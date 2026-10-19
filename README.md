# Autonomous Drone Tracking (PX4 SITL + MAVSDK + OpenCV)
 
A personal project to build a drone that can detect, track and eventually follow a target autonomously, developed entirely in simulation.
 
## Overview
 
- **Simulation**: PX4-Autopilot built from source, running in SITL with Gazebo (`x500_mono_cam` model, a quadrotor with an onboard camera), plus QGroundControl.
- **Flight control**: Python scripts using MAVSDK (asyncio) for connection, arming, takeoff, closed-loop altitude control from live telemetry, offboard position control (NED) and landing.
- **Vision**: the drone camera stream is received over UDP/RTP (H.264) through a GStreamer pipeline in OpenCV, then processed frame by frame.
## Vision pipeline
 
1. **Detection**: HSV color segmentation (red target, two hue ranges), morphological cleanup, largest contour as the target bounding box.
2. **Tracking**: OpenCV CSRT tracker initialized on the detection, with fallback to detection when the target is lost.
3. **Bounding-box correction**: every few frames, a local re-detection around the tracked box checks whether CSRT is stuck on a sub-part of the target, and re-initializes it at the true size if needed.
4. **State estimation**: a 2D constant-velocity Kalman filter (`cv2.KalmanFilter`, state = position + velocity in pixels) smooths the measurements and predicts the target motion.
## Files
 
| File | Purpose |
|---|---|
| `premier_test.py` | MAVSDK flight script: connect, arm, takeoff, altitude loop, offboard position control, land |
| `voir_camera.py` | Display the drone camera stream (GStreamer → OpenCV) |
| `detection_sphere.py` | HSV color detection of the red target |
| `tracking_csrt.py` | Detection + CSRT tracking with bounding-box correction |
| `tracking_kalman.py` | CSRT tracking + Kalman filter (position and velocity estimate) |
| `lancer_simu.sh` | Reminder of the commands to launch the simulation |
 
## Running
 
Requirements: Ubuntu 22.04 (WSL2 works), PX4-Autopilot, Gazebo, Python 3 with `mavsdk`, `opencv-python` built with GStreamer support, and `numpy`.
 
```bash
# Terminal 1: PX4 SITL + Gazebo with a camera drone
cd ~/PX4-Autopilot
HEADLESS=1 make px4_sitl gz_x500_mono_cam
 
# Terminal 2 (optional): QGroundControl
./QGroundControl.AppImage
 
# Terminal 3: flight script or vision scripts
python3 premier_test.py
python3 tracking_kalman.py
```
 
## Next steps
 
- Close the loop: convert the Kalman estimate of the target position into offboard velocity commands so the drone follows the target.
- Replace color detection with a learned detector (e.g. YOLO) for real-world targets.
- Test robustness to occlusions and fast target motion.
 
