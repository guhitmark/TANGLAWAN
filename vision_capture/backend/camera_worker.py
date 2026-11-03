import math
import threading
import time
import cv2
import numpy as np
import os

os.environ["OPENCV_LOG_LEVEL"] = "SILENT"

TIMING_PRINT = True   # prints a [WORKER] line every 60 frames per eye; set False to silence

class CameraWorker(threading.Thread):

    def __init__(self, eye, source_index, state, cap=None):
        super().__init__(daemon=True)
        self.eye          = eye
        self.state        = state
        self.source_index = source_index
        self.cap          = cap         
        self._running     = True
        self._lock        = threading.Lock()
        self._raw_jpeg    = b''
        self._crop_jpeg   = b''
        self._last_frame_time = time.perf_counter()
        self._fail_count  = 0
        self._ui_frame_counter = 0

        # loop timing: read, prep (crop+CLAHE+warp), track (+draw), record, encode
        self._tm   = [0.0, 0.0, 0.0, 0.0, 0.0]
        self._tm_n = 0
        self._tm_t0 = time.perf_counter()

        # Gentle CLAHE with large 4x4 tiles so it doesn't wash out the pupil
        self._clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(4, 4))
        
        self._setup_camera(self.cap)

    def _setup_camera(self, cap_obj):
        if cap_obj is not None and cap_obj.isOpened():
            os.system(f"v4l2-ctl -d /dev/video{self.source_index} -c exposure_auto=1 2>/dev/null")
            os.system(f"v4l2-ctl -d /dev/video{self.source_index} -c auto_exposure=1 2>/dev/null")
            os.system(f"v4l2-ctl -d /dev/video{self.source_index} -c exposure_auto_priority=0 2>/dev/null")
            os.system(f"v4l2-ctl -d /dev/video{self.source_index} -c exposure_absolute=5 2>/dev/null")
            os.system(f"v4l2-ctl -d /dev/video{self.source_index} -c exposure_time_absolute=15 2>/dev/null")
            os.system(f"v4l2-ctl -d /dev/video{self.source_index} -c gain=1 2>/dev/null")

    def _enhance_illumination(self, bgr_frame):
        # Works on ONE channel (the OV9281 is mono, all three BGR channels are equal)
        gray = np.ascontiguousarray(bgr_frame[:, :, 0]) if bgr_frame.ndim == 3 else bgr_frame
        even_gray = self._clahe.apply(gray)
        # Blend 75% original with 25% shadow-lifted to soften dark spots without overexposing
        return cv2.addWeighted(gray, 0.75, even_gray, 0.25, 0)

    def reconnect(self, source_index):
        with self._lock:
            if self.cap is not None:
                try: self.cap.release()
                except Exception: pass
                self.cap = None
            self.source_index = source_index

    def set_cap(self, cap):
        with self._lock:
            if self.cap is not None:
                try: self.cap.release()
                except Exception: pass
            self.cap = cap
            self._setup_camera(self.cap)

    def _apply_transform(self, frame):
        t = self.state.transform[self.eye]
        if t["flip_h"]: frame = cv2.flip(frame, 1)
        if t["flip_v"]: frame = cv2.flip(frame, 0)
        r = t["rotate"]
        if r == 1: frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif r == 2: frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif r == 3: frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return frame

    def _warp(self, sq_frame, original_side):
        scale = original_side / 400.0
        pts = np.float32(self.state.crop_points[self.eye]) * scale
        
        CROP_SIZE = 400 
        matrix = cv2.getPerspectiveTransform(
            pts,
            np.float32([[0, 0], [CROP_SIZE, 0],
                        [CROP_SIZE, CROP_SIZE], [0, CROP_SIZE]]),
        )
        return cv2.warpPerspective(sq_frame, matrix, (CROP_SIZE, CROP_SIZE), flags=cv2.INTER_NEAREST)

    def _release_cap(self):
        with self._lock:
            if self.cap is not None:
                try: self.cap.release()
                except Exception: pass
                self.cap = None

    def run(self):
        try:   # name the OS thread so profilers (top -H, jetson_profiler.py) show "cam-L" / "cam-R"
            import ctypes
            ctypes.CDLL("libc.so.6").prctl(15, ("cam-" + self.eye).encode(), 0, 0, 0)
        except Exception:
            pass
        target_interval = 1.0 / 30.0 
        
        # --- FIXED: Use high-resolution hardware counters for perfect timing ---
        self._last_frame_time = time.perf_counter()
        next_frame_time = time.perf_counter()
        self.state.live_fps[self.eye] = 30.0

        while self._running:
            client_active = (time.time() - self.state.last_client_ping) < 30.0
            enabled = self.state.left_enabled if self.eye == "L" else self.state.right_enabled

            if not enabled or not client_active:
                with self._lock:
                    self._raw_jpeg  = b''
                    self._crop_jpeg = b''
                time.sleep(0.3)
                next_frame_time = time.perf_counter() # Reset anchor while paused
                self._last_frame_time = next_frame_time
                continue

            with self._lock:
                cap_obj = self.cap

            if cap_obj is None or not cap_obj.isOpened():
                time.sleep(0.5)
                next_frame_time = time.perf_counter()
                self._last_frame_time = next_frame_time
                continue

            t_a = time.perf_counter()
            try:
                ret, frame = cap_obj.read()
            except Exception:
                ret, frame = False, None

            if not ret or frame is None or frame.size == 0:
                self._fail_count += 1
                if self._fail_count >= 10:
                    self._release_cap()
                    self._fail_count = 0
                    time.sleep(1.0)
                else:
                    time.sleep(0.05)
                next_frame_time = time.perf_counter()
                self._last_frame_time = next_frame_time
                continue

            self._fail_count = 0
            t_b = time.perf_counter()
            
            # --- HIGH PRECISION FPS CALCULATION ---
            now = time.perf_counter()
            dt  = now - self._last_frame_time
            self._last_frame_time = now
            if dt > 0:
                inst_fps = 1.0 / dt
                raw_fps = self.state.live_fps[self.eye] * 0.85 + inst_fps * 0.15
                # Snap minor sub-millisecond OS jitter within the 30Hz band to a steady 30.0
                if 28.5 <= raw_fps <= 31.5:
                    self.state.live_fps[self.eye] = 30.0
                else:
                    self.state.live_fps[self.eye] = round(raw_fps, 1)

            h, w = frame.shape[:2]
            side = min(h, w)
            x0   = (w - side) // 2
            sq   = frame[0:side, x0:x0 + side]

            # Subtle shadow balancing without washing out highlights
            sq_even = self._enhance_illumination(sq)           # 1 channel from here on

            base_gray = self._apply_transform(sq_even)
            warped_gray = self._warp(base_gray, original_side=side)
            warped = cv2.cvtColor(warped_gray, cv2.COLOR_GRAY2BGR)   # one conversion, for draw / recording
            t_c = time.perf_counter()

            result = self.state.pupil.process_frame(warped, eye=self.eye)

            if getattr(self.state, 'tracker_mode', False):
                display = warped.copy()
            else:
                display = self.state.pupil.draw(warped, result, eye=self.eye)

            if not math.isnan(result.diameter_mm):
                self.state.live_diameter_mm[self.eye] = result.diameter_mm
            self.state.live_confidence[self.eye] = result.confidence
            t_d = time.perf_counter()

            # ---------------------------------------------------------
            # FRAME CAPTURE LOGIC
            # ---------------------------------------------------------
            if self.state.is_recording and not self.state.is_paused:
                with self.state.lock:
                    if self.state.is_recording:
                        exact_elapsed = time.time() - self.state.start_time
                        
                        if getattr(self.state, 'tracker_mode', False):
                            active_time = min(exact_elapsed, 60.0)
                        else:
                            # UPDATED TIMELINE FOR 3-MINUTE PROTOCOL
                            if exact_elapsed <= 60:
                                active_time = exact_elapsed
                            elif exact_elapsed >= 120:
                                # Resumes recording after the 1-minute dark readaptation (60s to 120s)
                                active_time = 60 + min(exact_elapsed - 120, 60.0)
                            else:
                                # Freeze expected frames at 60.0 during dark adaptation so the video pauses
                                active_time = 60.0 
                                
                        if active_time >= 0:
                            expected_frames = int(active_time * 30.0)
                            frames_captured = self.state.frames_captured_L if self.eye == "L" else self.state.frames_captured_R
                            writer = self.state.writer_L if self.eye == "L" else self.state.writer_R
                            
                            if writer is not None:
                                while frames_captured < expected_frames:
                                    writer.write(display)
                                    frames_captured += 1
                                    if self.eye == "L": self.state.frames_captured_L = frames_captured
                                    else:               self.state.frames_captured_R = frames_captured

                        self.state.session_data[self.eye].append({
                            "t":     exact_elapsed,
                            "y":     result.diameter_mm,
                            "valid": result.is_valid,
                        })
                        
            elif not self.state.is_recording:
                with self.state.lock:
                    if self.eye == "L" and self.state.writer_L is not None:
                        self.state.writer_L.release()
                        self.state.writer_L = None
                    if self.eye == "R" and self.state.writer_R is not None:
                        self.state.writer_R.release()
                        self.state.writer_R = None
            # ---------------------------------------------------------
            t_e = time.perf_counter()

            # Encode JPEGs for the web UI every 2nd frame to save CPU cycles for 30 FPS tracking
            self._ui_frame_counter += 1
            if self._ui_frame_counter % 2 == 0:
                raw_disp_web = cv2.resize(base_gray, (400, 400))
                ok, buf  = cv2.imencode(".jpg", raw_disp_web, [cv2.IMWRITE_JPEG_QUALITY, 35])
                if ok:
                    with self._lock: self._raw_jpeg = buf.tobytes()

                ok, buf = cv2.imencode(".jpg", display, [cv2.IMWRITE_JPEG_QUALITY, 40])
                if ok:
                    with self._lock: self._crop_jpeg = buf.tobytes()
            t_f = time.perf_counter()

            # --- loop timing: where does each frame's time go? ---
            self._tm[0] += t_b - t_a
            self._tm[1] += t_c - t_b
            self._tm[2] += t_d - t_c
            self._tm[3] += t_e - t_d
            self._tm[4] += t_f - t_e
            self._tm_n += 1
            if self._tm_n >= 60:
                if TIMING_PRINT:
                    n = float(self._tm_n)
                    loop_ms = 1000.0 * (t_f - self._tm_t0) / n
                    print("[WORKER %s] loop %.1f ms (%.1f fps) | read %.1f | prep %.1f | track %.1f | record %.1f | encode %.1f"
                          % (self.eye, loop_ms, 1000.0 / loop_ms,
                             1000.0 * self._tm[0] / n, 1000.0 * self._tm[1] / n,
                             1000.0 * self._tm[2] / n, 1000.0 * self._tm[3] / n,
                             1000.0 * self._tm[4] / n))
                self._tm = [0.0, 0.0, 0.0, 0.0, 0.0]
                self._tm_n = 0
                self._tm_t0 = t_f

            # --- FIXED: ABSOLUTE TIMING ANCHOR ---
            next_frame_time += target_interval
            sleep_time = next_frame_time - time.perf_counter()
            
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                # If a frame took longer than 33ms to process (e.g. heavy CPU load),
                # immediately proceed and reset the anchor so we don't speed up to "catch up"
                next_frame_time = time.perf_counter()

    def latest_raw_jpeg(self):
        with self._lock: return self._raw_jpeg

    def latest_crop_jpeg(self):
        with self._lock: return self._crop_jpeg

    def stop(self):
        self._running = False
        self._release_cap()