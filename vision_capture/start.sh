#!/bin/bash
cd /home/tanglawan/Desktop/vision_capture_v1/vision_capture

echo "Unlocking Jetson Nano MAXN Performance Mode..."
sudo nvpmodel -m 0
sudo jetson_clocks

echo "Cleaning up previous sessions..."
sudo pkill -9 -f "python3 run.py" 2>/dev/null
sudo fuser -k /dev/gpiochip0 /dev/gpiochip1 2>/dev/null
sleep 1

python3 -c "import Jetson.GPIO as GPIO; GPIO.cleanup()" 2>/dev/null
sleep 0.5

echo "Starting TANGLAWAN server..."
source venv/bin/activate
python3 run.py