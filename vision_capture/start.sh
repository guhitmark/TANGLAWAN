#!/bin/bash
cd /home/tanglawan/Desktop/vision_capture_v1/vision_capture || exit 1

if [ -z "$INVOCATION_ID" ]; then
    # Runs only if started manually from a terminal
    echo "Unlocking Jetson Nano MAXN Performance Mode..."
    sudo nvpmodel -m 0
    sudo jetson_clocks
    echo "Cleaning up previous sessions..."
    sudo systemctl stop tanglawan.service 2>/dev/null
    sudo pkill -9 -f "[p]ython3 run.py" 2>/dev/null
    sudo fuser -k /dev/gpiochip0 /dev/gpiochip1 2>/dev/null
    sleep 1
fi

python3 -c "import Jetson.GPIO as GPIO; GPIO.cleanup()" 2>/dev/null
sleep 0.5

echo "Starting TANGLAWAN server..."
source venv/bin/activate
exec python3 run.py