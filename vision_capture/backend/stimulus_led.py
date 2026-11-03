import threading
import time

try:
    import Jetson.GPIO as GPIO
    HAS_GPIO = True
except ImportError:
    HAS_GPIO = False

from .gpio_manager import gpio_ready

# ── Hardware Definitions ───────────────────────────────────────────
PINS = {
    'L_BLUE': 13,
    'L_RED':  11,
    'R_BLUE': 16,
    'R_RED':  18,
}

SW_PWM_FREQ = 300
DUTY = {
    'L_BLUE': 100,
    'R_BLUE': 79,
    'L_RED':  100,
    'R_RED':  64,
}

RED_KEYS  = ('L_RED',  'R_RED')
BLUE_KEYS = ('L_BLUE', 'R_BLUE')

# ── Protocol Timings ───────────────────────────────────────────────
# Clinical Diagnostic Timeline
CLINICAL_RED_PULSE_AT_S  = 10
# UPDATED: Blue pulse now occurs at 130 seconds (10s baseline + 50s PIPR + 60s readapt + 10s baseline)
CLINICAL_BLUE_PULSE_AT_S = 130

# Tracker Dataset (1-Minute) Timeline
TRACKER_RED_PULSE_AT_S   = 10
TRACKER_BLUE_PULSE_AT_S  = 40

PULSE_DURATION_S = 1.0
TICK_S           = 0.05

# ── PWM Engine ─────────────────────────────────────────────────────
class SoftwarePWM:
    def __init__(self, pin, freq=100):
        self.pin = pin
        self.freq = freq
        self.duty = 0
        self.running = False
        self.thread = None
        if HAS_GPIO:
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
        if HAS_GPIO:
            GPIO.output(self.pin, GPIO.LOW)

    def _run(self):
        period = 1.0 / self.freq
        current_state = None 
        
        while self.running:
            d = self.duty
            if d == 0:
                if HAS_GPIO and current_state != 0:
                    GPIO.output(self.pin, GPIO.LOW)
                    current_state = 0
                time.sleep(0.05) 
                
            elif d == 100:
                if HAS_GPIO and current_state != 100:
                    GPIO.output(self.pin, GPIO.HIGH)
                    current_state = 100
                time.sleep(0.05)
                
            else:
                current_state = -1 
                on_time = period * (d / 100.0)
                off_time = period - on_time
                if HAS_GPIO:
                    GPIO.output(self.pin, GPIO.HIGH)
                    time.sleep(on_time)
                    GPIO.output(self.pin, GPIO.LOW)
                    time.sleep(off_time)


# ── Main Controller ────────────────────────────────────────────────
class StimulusController(threading.Thread):
    def __init__(self, state):
        super().__init__(daemon=True, name="stimulus-leds")
        self.state = state
        self._running    = True
        self._gpio_ready = False
        self._pwms       = {}

    def run(self):
        if HAS_GPIO:
            gpio_ready.wait(timeout=10.0)
            try:
                if GPIO.getmode() is None:
                    GPIO.setmode(GPIO.BOARD)
                GPIO.setwarnings(False)
                
                for key, pin in PINS.items():
                    self._pwms[key] = SoftwarePWM(pin, SW_PWM_FREQ)
                    self._pwms[key].start(0)
                self._gpio_ready = True
                print("[StimulusLED] GPIO ready - Red/Blue LEDs initialized LOW")
            except Exception as e:
                print(f"[StimulusLED] GPIO init failed -- LEDs disabled: {e}")

        was_recording = False
        red_fired     = False
        blue_fired    = False
        
        # New: Track exactly when to turn the LEDs off without spawning threads
        red_off_time  = 0.0
        blue_off_time = 0.0

        while self._running:
            now = time.time()
            
            if getattr(self.state, 'is_recording', False):
                
                # Reset fire flags at the start of a new recording sequence
                if not was_recording:
                    red_fired     = False
                    blue_fired    = False
                    red_off_time  = 0.0
                    blue_off_time = 0.0
                    was_recording = True

                elapsed_s = int(now - self.state.start_time)
                is_tracker = getattr(self.state, 'tracker_mode', False)

                # --- TRACKER DATASET PROTOCOL (1 MINUTE) ---
                if is_tracker:
                    if not red_fired and elapsed_s >= TRACKER_RED_PULSE_AT_S:
                        red_fired = True
                        self._set(RED_KEYS, True)
                        red_off_time = now + PULSE_DURATION_S

                    if not blue_fired and elapsed_s >= TRACKER_BLUE_PULSE_AT_S:
                        blue_fired = True
                        self._set(BLUE_KEYS, True)
                        blue_off_time = now + PULSE_DURATION_S

                # --- CLINICAL DIAGNOSTIC PROTOCOL (180 SECONDS) ---
                else:
                    if not red_fired and elapsed_s >= CLINICAL_RED_PULSE_AT_S:
                        red_fired = True
                        self._set(RED_KEYS, True)
                        red_off_time = now + PULSE_DURATION_S

                    if not blue_fired and elapsed_s >= CLINICAL_BLUE_PULSE_AT_S:
                        blue_fired = True
                        self._set(BLUE_KEYS, True)
                        blue_off_time = now + PULSE_DURATION_S
                        
            else:
                was_recording = False

            # --- NON-BLOCKING SHUTOFF LOGIC ---
            # The loop naturally checks if it's time to turn the LEDs off 
            # without ever pausing or spawning a heavy Thread.
            if red_off_time > 0 and now >= red_off_time:
                self._set(RED_KEYS, False)
                red_off_time = 0.0
                
            if blue_off_time > 0 and now >= blue_off_time:
                self._set(BLUE_KEYS, False)
                blue_off_time = 0.0

            time.sleep(TICK_S)

    def _set(self, keys, on):
        if self._gpio_ready:
            for key in keys:
                self._pwms[key].ChangeDutyCycle(DUTY[key] if on else 0)

    def stop(self):
        self._running = False
        self._set(PINS.keys(), False)
        for pwm in self._pwms.values():
            pwm.stop()
        self._gpio_ready = False