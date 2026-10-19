# Terminal 1
cd ~/PX4-Autopilot
HEADLESS=1 make px4_sitl gz_x500_mono_cam

# Terminal 2
cd ~
export QT_QPA_PLATFORM=xcb
./QGroundControl.AppImage

# Terminal 3
cd ~/drone-tracking
python3 premier_test.py
