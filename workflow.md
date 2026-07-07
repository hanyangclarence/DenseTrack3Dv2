```
(base) labeng@wilm-rob-06:~/yanghan/code/hardwares/zed-sdk$ python capture_rgbd.py --viz_min_m 0 --viz_max_m 0.5

(track4world) labeng@wilm-rob-06:~/yanghan/code/vision/Track4World$ python scripts/run_dino_sam2.py     --video-path /home/labeng/yanghan/code/hardwares/zed-sdk/rgbd_capture/color.mp4     --sam2-checkpoint checkpoints/sam2.1_hiera_large.pt     --output-dir results/measure     --text-prompt "red tape measure."

(densetrack3d) labeng@wilm-rob-06:~/yanghan/code/vision/DenseTrack3Dv2$ python3 preprocess/track_windowed.py   --video /home/labeng/yanghan/code/hardwares/zed-sdk/rgbd_capture/color.mp4   --depth /home/labeng/yanghan/code/hardwares/zed-sdk/rgbd_capture/depth   --mask-dir /home/labeng/yanghan/code/vision/Track4World/results/measure/mask   --output-path results/zed_windowed   --start-frame 0 --num-frames 470 --win 20 --stride 5 --grid-size 40

(densetrack3d) labeng@wilm-rob-06:~/yanghan/code/vision/DenseTrack3Dv2$ python3 visualizer/vis_densetrack3d_trails.py --filepath results/zed_windowed/dense_3d_track.pkl  --smooth_sigma 3.0
```