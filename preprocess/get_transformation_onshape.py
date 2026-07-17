import os, hmac, hashlib, base64, secrets, json
from email.utils import formatdate
from urllib.parse import urlparse
import requests
import numpy as np

# Onshape: x=up, y=right, z=front
# Genesis: x=front, y=left, z=up
# Mapping: x_o->z_g, y_o->-y_g, z_o->x_g
R_ONSHAPE_TO_GENESIS = np.array([
    [0,  0, 1],
    [0, -1, 0],
    [1,  0, 0],
], dtype=float)

def convert_T_onshape_to_genesis(T_onshape: np.ndarray) -> np.ndarray:
    """Convert a 4x4 transform from Onshape coords to Genesis coords via conjugation."""
    R4 = np.eye(4)
    R4[:3, :3] = R_ONSHAPE_TO_GENESIS
    R4_inv = np.eye(4)
    R4_inv[:3, :3] = R_ONSHAPE_TO_GENESIS.T
    return T_onshape @ R4_inv

# ======================
# Config
# ======================
BASE = "https://cad.onshape.com"

did = "9ed8a256ac83f2e5e1899d42"
wid = "d9577fa8a24f455cda572931"
eid = "6934d1d097bd9fdf6d6fd52d"

# Recommended: use environment variables
ACCESS_KEY = os.environ.get("ONSHAPE_ACCESS_KEY", "on_qH2yo7Hg2uIYpTcgdpEe7")
SECRET_KEY = os.environ.get("ONSHAPE_SECRET_KEY", "sFnzNE3KHGEPIkK0IM4MnsccWLBG7puMkcEvs6vdGpbiK4TV")
print("ACCESS_KEY:", ACCESS_KEY)
print("SECRET_KEY:", SECRET_KEY)

# === Part Studio / Mate Connector feature IDs ===
QUEST_PS_EID = "817bb53830f6db5ed929262e"
QUEST_MC_FEATURE_ID = "FSWGcatNmTO5jTL_52"

CTRL_PS_EID = "72195f9688d2d54442439526"
CTRL_MC_FEATURE_ID = "FkM3YOLyy3VsSGc_50"

# (Optional) If a Part Studio has multiple parts, specify with partId
# From logs, Quest3 partId was JFH
QUEST_PART_ID = None
# Fill in controller if needed (e.g., "JMD")
CTRL_PART_ID = None

# ======================
# Auth (HMAC)
# ======================
def make_onshape_headers(method: str, full_url: str, content_type: str = ""):
    auth_date = formatdate(timeval=None, localtime=False, usegmt=True)
    nonce = secrets.token_hex(16)

    u = urlparse(full_url)
    path = u.path
    query = u.query or ""

    canonical = (
        method.upper() + "\n" +
        nonce + "\n" +
        auth_date + "\n" +
        (content_type or "") + "\n" +
        path + "\n" +
        query + "\n"
    ).lower()

    if isinstance(SECRET_KEY, str):
        secret_bytes = SECRET_KEY.encode("utf-8")
    else:
        secret_bytes = SECRET_KEY

    sig = base64.b64encode(
        hmac.new(secret_bytes, canonical.encode("utf-8"), hashlib.sha256).digest()
    ).decode("ascii")

    authorization = f"On {ACCESS_KEY}:HmacSHA256:{sig}"
    headers = {
        "Date": auth_date,
        "On-Nonce": nonce,
        "Authorization": authorization,
    }
    if content_type:
        headers["Content-Type"] = content_type
    return headers

def _get_json(url: str):
    headers = make_onshape_headers("GET", url)
    r = requests.get(url, headers=headers)
    r.raise_for_status()
    return r.json()

def _post_json(url: str, body: dict):
    headers = make_onshape_headers("POST", url, content_type="application/json")
    r = requests.post(url, headers=headers, json=body)
    r.raise_for_status()
    return r.json()

# ======================
# Assembly (v12) fetch
# ======================
def get_assembly_v12(did: str, wid: str, eid: str,
                    include_mate_features: bool = False,
                    include_non_solids: bool = True,
                    include_mate_connectors: bool = False,
                    exclude_suppressed: bool = False):
    url = (
        f"{BASE}/api/v12/assemblies/d/{did}/w/{wid}/e/{eid}"
        f"?includeMateFeatures={'true' if include_mate_features else 'false'}"
        f"&includeNonSolids={'true' if include_non_solids else 'false'}"
        f"&includeMateConnectors={'true' if include_mate_connectors else 'false'}"
        f"&excludeSuppressed={'true' if exclude_suppressed else 'false'}"
    )
    return _get_json(url)

# ======================
# Part Studio features
# ======================
def get_partstudio_features(did: str, wid: str, ps_eid: str):
    url = f"{BASE}/api/partstudios/d/{did}/w/{wid}/e/{ps_eid}/features"
    return _get_json(url)

# ======================
# FeatureScript eval helpers
# ======================
def _fs_unwrap(node):
    """Unwrap eval FeatureScript response (result.message.value) to Python values."""
    if isinstance(node, dict):
        if "message" in node and isinstance(node["message"], dict):
            msg = node["message"]
            if "value" in msg and len(msg) == 1:
                return msg["value"]
            if "key" in msg and "value" in msg:
                return (_fs_unwrap(msg["key"]), _fs_unwrap(msg["value"]))
            return {k: _fs_unwrap(v) for k, v in msg.items()}
        return {k: _fs_unwrap(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_fs_unwrap(x) for x in node]
    return node

def eval_featurescript(did: str, wid: str, ps_eid: str, script: str, library_version: int = 2837):
    endpoints = [
        f"{BASE}/api/partstudios/d/{did}/w/{wid}/e/{ps_eid}/featurescript?rollbackBarIndex=-1",
        f"{BASE}/api/v6/partstudios/d/{did}/w/{wid}/e/{ps_eid}/featurescript?rollbackBarIndex=-1",
        f"{BASE}/api/v8/partstudios/d/{did}/w/{wid}/e/{ps_eid}/featurescript?rollbackBarIndex=-1",
    ]
    body = {"libraryVersion": library_version, "script": script, "queries": []}

    last_err = None
    for url in endpoints:
        try:
            return _post_json(url, body)
        except requests.HTTPError as e:
            last_err = (url, e.response.status_code, e.response.text[:500])
            continue
    raise RuntimeError(f"evalFeatureScript failed: {last_err}")

def get_mateconnector_T_partstudio(did: str, wid: str, ps_eid: str, mc_feature_id: str, library_version: int = 2837) -> np.ndarray:
    fs = f'''
function(context is Context, definition is map)
{{
    const mc = evMateConnector(context, {{
        "mateConnector" : qBodyType(qCreatedBy(makeId("{mc_feature_id}"), EntityType.BODY), BodyType.MATE_CONNECTOR)
    }});
    const T = toWorld(mc);
    return [
      T.linear[0][0], T.linear[0][1], T.linear[0][2], T.translation[0] / meter,
      T.linear[1][0], T.linear[1][1], T.linear[1][2], T.translation[1] / meter,
      T.linear[2][0], T.linear[2][1], T.linear[2][2], T.translation[2] / meter,
      0, 0, 0, 1
    ];
}}
'''
    out = eval_featurescript(did, wid, ps_eid, fs, library_version=library_version)
    raw = out.get("result", {}).get("message", {}).get("value")
    if raw is None:
        print("FS response result:", json.dumps(out.get("result"), indent=2))
        print("FS response notices:", json.dumps(out.get("notices"), indent=2))
        raise KeyError(f"Unexpected evalFeatureScript response keys: {list(out.keys())}")

    val = _fs_unwrap(raw)

    if isinstance(val, list) and len(val) == 16 and isinstance(val[0], dict) and "value" in val[0]:
        flat = [x["value"] for x in val]
    elif isinstance(val, list) and len(val) == 16 and isinstance(val[0], (int, float)):
        flat = val
    else:
        raise ValueError(f"Unexpected FS return: type={type(val)}, len={len(val) if isinstance(val, list) else None}")

    return np.array(flat, dtype=float).reshape(4, 4)

# ======================
# Assembly transform utilities
# ======================
def as_mat4(T):
    if isinstance(T, list) and len(T) == 16:
        return np.array(T, dtype=float).reshape(4, 4)
    if isinstance(T, list) and len(T) == 4 and isinstance(T[0], list):
        return np.array(T, dtype=float)
    raise ValueError(f"Unknown transform format: {type(T)} / len={len(T) if isinstance(T,list) else None}")

def build_occ_transform_map(asm):
    m = {}
    for occ in asm["rootAssembly"].get("occurrences", []):
        path = occ.get("path")
        if not path:
            continue
        m[tuple(path)] = as_mat4(occ["transform"])
    return m

def instance_ids_for_element(asm, element_id: str, part_id: str | None = None):
    ids = []
    for inst in asm["rootAssembly"].get("instances", []):
        if inst.get("suppressed", False):
            continue
        if inst.get("elementId") != element_id:
            continue
        if part_id is not None and inst.get("partId") != part_id:
            continue
        ids.append(inst["id"])
    return ids

def pick_first_occ_path(occ_map, instance_ids):
    for iid in instance_ids:
        p = (iid,)  # leaf path
        if p in occ_map:
            return list(p)
    return None

def compute_world_mc(asm, occ_map, ps_eid: str, mc_feature_id: str, part_id: str | None = None):
    # 1) MC in PartStudio coordinates
    T_ps_mc = get_mateconnector_T_partstudio(did, wid, ps_eid, mc_feature_id)

    # 2) Choose a matching instance in the assembly
    inst_ids = instance_ids_for_element(asm, ps_eid, part_id=part_id)
    if not inst_ids:
        raise KeyError(f"No instances for elementId={ps_eid} partId={part_id} in this assembly")

    occ_path = pick_first_occ_path(occ_map, inst_ids)
    if occ_path is None:
        raise KeyError(f"No occurrence found for instance ids (first few)={inst_ids[:5]}")

    T_w_occ = occ_map[tuple(occ_path)]
    T_w_mc = T_w_occ @ T_ps_mc
    return T_w_mc, occ_path

# ======================
# Main
# ======================
def main():
    # 1) Assembly v12 fetch: Keep includeNonSolids=true for Quest3 to be visible
    asm = get_assembly_v12(
        did, wid, eid,
        include_mate_features=False,
        include_non_solids=True,
        include_mate_connectors=False,
        exclude_suppressed=False
    )

    ra = asm["rootAssembly"]
    print("instances:", len(ra.get("instances", [])))
    print("occurrences:", len(ra.get("occurrences", [])))

    # 2) Occurrence transform map
    occ_map = build_occ_transform_map(asm)

    # 3) Compute world mate connector transforms
    T_w_q, q_path = compute_world_mc(asm, occ_map, QUEST_PS_EID, QUEST_MC_FEATURE_ID, part_id=QUEST_PART_ID)
    T_w_c, c_path = compute_world_mc(asm, occ_map, CTRL_PS_EID, CTRL_MC_FEATURE_ID, part_id=CTRL_PART_ID)
    
    print("T_w_q (Onshape):\n", T_w_q)
    print("T_w_c (Onshape):\n", T_w_c)

    # 4) Relative transform (controller MC -> quest MC)
    T_q_to_c = np.linalg.inv(T_w_c) @ T_w_q
    T_c_to_q = np.linalg.inv(T_w_q) @ T_w_c

    print("\nQuest3 occurrence path:", q_path)
    print("Controller occurrence path:", c_path)
    print("\nT_controller_to_quest3 (Onshape):\n", T_c_to_q)
    print("\nT_quest3_to_controller (Onshape):\n", T_q_to_c)

    translation = T_c_to_q[:3, 3]
    translation_length = np.linalg.norm(translation)
    print("translation length:", translation_length)

    # 5) Convert to Genesis coordinate system
    T_c_to_q_gen = convert_T_onshape_to_genesis(T_c_to_q)

    print("\n=== Genesis coordinates (x=front, y=left, z=up) ===")
    print("T_controller_to_quest3 (Genesis):\n", T_c_to_q_gen)
    print("translation (Genesis):", T_c_to_q_gen[:3, 3])
    print("translation length (Genesis):", np.linalg.norm(T_c_to_q_gen[:3, 3]))

if __name__ == "__main__":
    main()