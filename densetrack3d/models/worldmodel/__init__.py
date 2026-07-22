"""Object-flow intent world model (spec 2026-07-20-object-flow-model-architecture-detail)."""
from densetrack3d.models.worldmodel.intent_model import (
    IntentModel,
    IntentModelConfig,
    intent_loss,
)

__all__ = ["IntentModel", "IntentModelConfig", "intent_loss"]
