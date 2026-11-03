"""
led_spectro_test.py
TANGLAWAN LED Test — Interactive Spectroradiometer Calibration
"""

import time
import threading
import Jetson.GPIO as GPIO
import sys

class SoftwarePWM:
    def __init__(self, pin, freq=100):
        self.pin = pin
        self.freq = freq
        self.duty = 0
        self.running = False
        self.thread = None
        GPIO.setup(self.pin, GPIO.OUT, initial=GPIO.LOW)

    def start(self, duty):
        self.duty = max(0.0, min(100.0, float(duty)))
        self.running = True
        self.thread = threading.Thread(target=self._run)
        self.thread.daemon = True
        self.thread.start()

    def ChangeDutyCycle(self, duty):
        self.duty = max(0.0, min(100.0, float(duty)))

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        GPIO.output(self.pin, GPIO.LOW)

    def _run(self):
        period = 1.0 / self.freq
        while self.running:
            d = self.duty
            if d == 0:
                GPIO.output(self.pin, GPIO.LOW)
                time.sleep(0.01)
            elif d == 100:
                GPIO.output(self.pin, GPIO.HIGH)
                time.sleep(0.01)
            else:
                on_time = period * (d / 100.0)
                t0 = time.perf_counter()
                GPIO.output(self.pin, GPIO.HIGH)
                while (time.perf_counter() - t0) < on_time:
                    pass
                GPIO.output(self.pin, GPIO.LOW)
                while (time.perf_counter() - t0) < period:
                    pass

# ── Pin Definitions ────────────────────────────────────────────────
L_BLUE_PIN = 13
L_RED_PIN  = 11

R_BLUE_PIN = 16
R_RED_PIN  = 18

# ── Duty Cycle Tuning ──────────────────────────────────────────────
SW_PWM_FREQ = 100

L_BLUE_DUTY = 100
R_BLUE_DUTY = 90

L_RED_DUTY  = 100
R_RED_DUTY  = 80

# ── Setup ──────────────────────────────────────────────────────────
GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)

pwms = {
    'L_BLUE': SoftwarePWM(L_BLUE_PIN, SW_PWM_FREQ),
    'L_RED':  SoftwarePWM(L_RED_PIN,  SW_PWM_FREQ),
    'R_BLUE': SoftwarePWM(R_BLUE_PIN, SW_PWM_FREQ),
    'R_RED':  SoftwarePWM(R_RED_PIN,  SW_PWM_FREQ),
}

for pwm in pwms.values():
    pwm.start(0)

def set_color_state(color):
    """Turns the requested color ON and ensures the other is OFF."""
    if color == 'all':
        pwms['L_BLUE'].ChangeDutyCycle(L_BLUE_DUTY)
        pwms['R_BLUE'].ChangeDutyCycle(R_BLUE_DUTY)
        pwms['L_RED'].ChangeDutyCycle(L_RED_DUTY)
        pwms['R_RED'].ChangeDutyCycle(R_RED_DUTY)
        print("\n>> ALL LEDs are ON. (Waiting for spectroradiometer readings...)")
        print(">> Type 'x' and press Enter to turn off and return to menu.")

    elif color == 'red':
        pwms['L_BLUE'].ChangeDutyCycle(0)
        pwms['R_BLUE'].ChangeDutyCycle(0)
        pwms['L_RED'].ChangeDutyCycle(L_RED_DUTY)
        pwms['R_RED'].ChangeDutyCycle(R_RED_DUTY)
        print("\n>> RED LEDs are ON. (Waiting for spectroradiometer readings...)")
        print(">> Type 'x' and press Enter to turn off and return to menu.")
    
    elif color == 'blue':
        pwms['L_RED'].ChangeDutyCycle(0)
        pwms['R_RED'].ChangeDutyCycle(0)
        pwms['L_BLUE'].ChangeDutyCycle(L_BLUE_DUTY)
        pwms['R_BLUE'].ChangeDutyCycle(R_BLUE_DUTY)
        print("\n>> BLUE LEDs are ON. (Waiting for spectroradiometer readings...)")
        print(">> Type 'x' and press Enter to turn off and return to menu.")
        
    elif color == 'off':
        for pwm in pwms.values():
            pwm.ChangeDutyCycle(0)

# ── Interactive Menu ───────────────────────────────────────────────
print("==================================================")
print(" TANGLAWAN Spectroradiometer Calibration Mode")
print("==================================================")

try:
    # Trigger all LEDs ON immediately upon running
    set_color_state('all')
    while True:
        cancel = input().strip().lower()
        if cancel == 'x':
            break

    # Enter the standard interactive menu loop
    while True:
        set_color_state('off')
        print("\nOptions:")
        print("  [a] - Turn ON Continuous ALL (Red & Blue)")
        print("  [r] - Turn ON Continuous RED")
        print("  [b] - Turn ON Continuous BLUE")
        print("  [Ctrl+C] - Exit program")
        
        choice = input("Select an option: ").strip().lower()
        
        if choice == 'a':
            set_color_state('all')
            while True:
                cancel = input().strip().lower()
                if cancel == 'x':
                    break
        
        elif choice == 'r':
            set_color_state('red')
            while True:
                cancel = input().strip().lower()
                if cancel == 'x':
                    break
                    
        elif choice == 'b':
            set_color_state('blue')
            while True:
                cancel = input().strip().lower()
                if cancel == 'x':
                    break
        else:
            print("Invalid input. Please enter 'a', 'r', or 'b'.")

except KeyboardInterrupt:
    print("\n\n[TEST] Session aborted by user.")

finally:
    print("[TEST] Shutting down LEDs and cleaning up GPIO...")
    for pwm in pwms.values():
        pwm.stop()
    GPIO.cleanup()
    print("[TEST] Done.")
    sys.exit(0)