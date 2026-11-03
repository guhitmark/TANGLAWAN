"""
gpio_manager.py
---------------------------------------------------------------------
Single shared GPIO initializer. 
The stimulus_led module imports this and waits on gpio_ready 
before touching any pins to fire the Red/Blue light pulses.
---------------------------------------------------------------------
"""

import threading

# MASTER TOGGLE: Set to True only when physical hardware is connected and ready.
ENABLE_GPIO = True

try:
    import Jetson.GPIO as GPIO
    HAS_GPIO = True
except ImportError:
    HAS_GPIO = False

gpio_ready = threading.Event()

def init_gpio():
    """
    Initializes pin mode or safely bypasses hardware interaction if disabled.
    """
    if not HAS_GPIO or not ENABLE_GPIO:
        print("[GPIO] Hardware operations are DISABLED (Running in pure software mode)")
        gpio_ready.set()
        return
        
    try:
        GPIO.setmode(GPIO.BOARD)
        print("[GPIO] setmode(BOARD) done")
        gpio_ready.set()
    except Exception as e:
        print(f"[GPIO] setmode failed: {e}")
        gpio_ready.set()