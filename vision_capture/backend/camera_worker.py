"""TANGLAWAN binocular camera worker, 90-second continuous PLR protocol.

Clinical: record continuously 0–90 s, including 32–47 s washout.
Tracker: retain independent 60-second mode.
LED scheduling belongs to stimulus_led.py; automatic stop belongs to routes/status.py.

Video: 30 FPS MJPG AVI (time-based frame duplication compensates for late reads).
Telemetry: one sample per actual camera observation; do not fabricate pupil sizes.
Staged recordings are retained for explicit Save/Discard by SessionState.
"""

import math
import os
import threading
import time

import cv2
import numpy as np

os.environ["OPENCV_LOG_LEVEL"] = "SILENT"

TIMING_PRINT = True
OUTPUT_FPS = 30.0  # Keep synchronized with session_state.py VideoWriter TARGET_FPS
CLINICAL_DURATION_S = 90.0
TRACKER_DURATION_S = 60.0
LAG_WARNING_FRAMES = 15  # Warn when AVI timebase requires many repeated frames


class CameraWorker(threading.Thread):
    def __init__(self, eye, source_index, state, cap=None):
        super().__init__(daemon=True, name="cam-" + str(eye))
        self.eye = eye
        self.state = state
        self.source_index = source_index
        self.cap = cap
        self._running = True
        self._lock = threading.Lock()
        self._raw_jpeg = b""
        self._crop_jpeg = b""
        self._last_frame_time = time.perf_counter()
        self._fail_count = 0
        self._ui_frame_counter = 0
        self._tm = [0.0] * 5
        self._tm_n = 0
        self._tm_t0 = time.perf_counter()
        self._clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(4, 4))
        self._setup_camera(self.cap)

    def _setup_camera(self, cap_obj):
        if cap_obj is not None and cap_obj.isOpened() and os.name == "posix":
            for control in (
                "exposure_auto=1", "auto_exposure=1",
                "exposure_auto_priority=0", "exposure_absolute=5",
                "exposure_time_absolute=15", "gain=1",
            ):
                os.system(
                    "v4l2-ctl -d /dev/video{} -c {} 2>/dev/null".format(
                        self.source_index, control
                    )
                )

    def _enhance_illumination(self, bgr_frame):
        gray = (np.ascontiguousarray(bgr_frame[:, :, 0])
                if bgr_frame.ndim == 3 else np.ascontiguousarray(bgr_frame))
        enhanced = self._clahe.apply(gray)
        return cv2.addWeighted(gray, 0.75, enhanced, 0.25, 0)

    def reconnect(self, source_index):
        with self._lock:
            if self.cap is not None:
                try:
                    self.cap.release()
                except Exception:
                    pass
            self.cap = None
            self.source_index = source_index

    def set_cap(self, cap):
        with self._lock:
            if self.cap is not None and self.cap is not cap:
                try:
                    self.cap.release()
                except Exception:
                    pass
            self.cap = cap
            self._setup_camera(self.cap)

    def _apply_transform(self, frame):
        t = self.state.transform[self.eye]
        if t["flip_h"]:
            frame = cv2.flip(frame, 1)
        if t["flip_v"]:
            frame = cv2.flip(frame, 0)
        r = t["rotate"]
        if r == 1:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif r == 2:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif r == 3:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return frame

    def _warp(self, sq_frame, original_side):
        pts = np.float32(self.state.crop_points[self.eye]) * (original_side / 400.0)
        dst = np.float32([[0, 0], [400, 0], [400, 400], [0, 400]])
        matrix = cv2.getPerspectiveTransform(pts, dst)
        return cv2.warpPerspective(sq_frame, matrix, (400, 400), flags=cv2.INTER_NEAREST)

    def _release_cap(self):
        with self._lock:
            if self.cap is not None:
                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = None

    def _release_writer(self):
        """Only the owning eye worker releases its VideoWriter.

        Do not reset has_staged_data: the Save/Report UI needs it.
        """
        with self.state.lock:
            attr = "writer_L" if self.eye == "L" else "writer_R"
            writer = getattr(self.state, attr, None)
            if writer is not None:
                try:
                    writer.release()
                except Exception as exc:
                    print("[WORKER {}] AVI release failed: {}".format(self.eye, exc))
                setattr(self.state, attr, None)

    def _record(self, display, result):
        with self.state.lock:
            if not self.state.is_recording or self.state.is_paused:
                return
            elapsed = time.time() - self.state.start_time
            duration = (TRACKER_DURATION_S if getattr(self.state, "tracker_mode", False)
                        else CLINICAL_DURATION_S)
            # Capture *all* clinical phases, including the washout (32–47 s).
            # The status route stops the session at the duration; do not append
            # any observations beyond the protocol if a status poll runs late.
            if elapsed < 0.0 or elapsed > duration:
                return

            writer = self.state.writer_L if self.eye == "L" else self.state.writer_R
            count_attr = "frames_captured_L" if self.eye == "L" else "frames_captured_R"
            frames_written = getattr(self.state, count_attr)
            target = min(int(elapsed * OUTPUT_FPS), int(duration * OUTPUT_FPS))
            if writer is not None and target > frames_written:
                needed = target - frames_written
                if needed > LAG_WARNING_FRAMES:
                    # AVI frame count still tracks time, but repeated frames during
                    # a long read stall are not genuinely acquired measurements.
                    print("[WORKER {}] Camera lag: {} frame slots behind; filling AVI only".format(
                        self.eye, needed
                    ))
                # Preserve video timebase. Do not append repeated telemetry frames.
                for _ in range(needed):
                    writer.write(display)
                setattr(self.state, count_attr, target)

            try:
                diam = float(result.diameter_mm)
            except (TypeError, ValueError):
                diam = float("nan")
            is_valid = bool(result.is_valid) and math.isfinite(diam)
            self.state.session_data[self.eye].append({
                "t": float(elapsed), "y": diam, "valid": is_valid,
            })

    def run(self):
        try:
            import ctypes
            ctypes.CDLL("libc.so.6").prctl(15, ("cam-" + self.eye).encode(), 0, 0, 0)
        except Exception:
            pass
        target_interval = 1.0 / OUTPUT_FPS
        self._last_frame_time = time.perf_counter()
        next_frame_time = time.perf_counter()
        self.state.live_fps[self.eye] = OUTPUT_FPS

        try:
            while self._running:
                # This check must happen BEFORE skipping an unavailable/disabled
                # camera, otherwise the AVI writer can remain open on completion.
                if not self.state.is_recording:
                    self._release_writer()

                enabled = (self.state.left_enabled if self.eye == "L"
                           else self.state.right_enabled)
                client_active = (time.time() - self.state.last_client_ping) < 30.0
                # During a recording, never stop capture merely because the UI
                # lost its heartbeat. The 90-second protocol must remain intact.
                if not enabled or (not client_active and not self.state.is_recording):
                    with self._lock:
                        self._raw_jpeg = b""
                        self._crop_jpeg = b""
                    time.sleep(0.3)
                    next_frame_time = time.perf_counter()
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
                now = time.perf_counter()
                dt = now - self._last_frame_time
                self._last_frame_time = now
                if dt > 0:
                    inst_fps = 1.0 / dt
                    smooth_fps = self.state.live_fps[self.eye] * 0.85 + inst_fps * 0.15
                    self.state.live_fps[self.eye] = (
                        OUTPUT_FPS if 28.5 <= smooth_fps <= 31.5 else round(smooth_fps, 1)
                    )

                try:
                    h, w = frame.shape[:2]
                    side = min(h, w)
                    x0 = (w - side) // 2
                    sq = frame[:side, x0:x0 + side]
                    gray = self._enhance_illumination(sq)
                    base_gray = self._apply_transform(gray)
                    warped_gray = self._warp(base_gray, original_side=side)
                    warped = cv2.cvtColor(warped_gray, cv2.COLOR_GRAY2BGR)
                    t_c = time.perf_counter()
                    result = self.state.pupil.process_frame(warped, eye=self.eye)
                    if getattr(self.state, "tracker_mode", False):
                        display = warped.copy()
                    else:
                        display = self.state.pupil.draw(warped, result, eye=self.eye)
                    try:
                        diameter = float(result.diameter_mm)
                    except (TypeError, ValueError):
                        diameter = float("nan")
                    # If tracking is lost, don't leave a stale displayed pupil
                    # diameter as if it is a fresh measurement.
                    self.state.live_diameter_mm[self.eye] = (
                        diameter if result.is_valid and math.isfinite(diameter) else float("nan")
                    )
                    self.state.live_confidence[self.eye] = float(result.confidence)
                    t_d = time.perf_counter()
                    self._record(display, result)
                    t_e = time.perf_counter()

                    self._ui_frame_counter += 1
                    if self._ui_frame_counter % 2 == 0:
                        raw_web = cv2.resize(base_gray, (400, 400))
                        ok, buf = cv2.imencode(".jpg", raw_web, [cv2.IMWRITE_JPEG_QUALITY, 35])
                        if ok:
                            with self._lock:
                                self._raw_jpeg = buf.tobytes()
                        ok, buf = cv2.imencode(".jpg", display, [cv2.IMWRITE_JPEG_QUALITY, 40])
                        if ok:
                            with self._lock:
                                self._crop_jpeg = buf.tobytes()
                    t_f = time.perf_counter()
                except Exception as exc:
                    print("[WORKER {}] Processing error: {}".format(self.eye, exc))
                    next_frame_time = time.perf_counter()
                    continue

                self._tm[0] += t_b - t_a
                self._tm[1] += t_c - t_b
                self._tm[2] += t_d - t_c
                self._tm[3] += t_e - t_d
                self._tm[4] += t_f - t_e
                self._tm_n += 1
                if self._tm_n >= 60:
                    if TIMING_PRINT:
                        n = float(self._tm_n)
                        loop_ms = max(0.001, 1000.0 * (t_f - self._tm_t0) / n)
                        print("[WORKER %s] loop %.1f ms (%.1f fps) | read %.1f | prep %.1f | track %.1f | record %.1f | encode %.1f" % (
                            self.eye, loop_ms, 1000.0 / loop_ms,
                            1000.0 * self._tm[0] / n, 1000.0 * self._tm[1] / n,
                            1000.0 * self._tm[2] / n, 1000.0 * self._tm[3] / n,
                            1000.0 * self._tm[4] / n,
                        ))
                    self._tm = [0.0] * 5
                    self._tm_n = 0
                    self._tm_t0 = t_f

                next_frame_time += target_interval
                sleep_s = next_frame_time - time.perf_counter()
                if sleep_s > 0:
                    time.sleep(sleep_s)
                else:
                    next_frame_time = time.perf_counter()
        finally:
            self._release_writer()
            self._release_cap()

    def latest_raw_jpeg(self):
        with self._lock:
            return self._raw_jpeg

    def latest_crop_jpeg(self):
        with self._lock:
            return self._crop_jpeg

    def stop(self):
        self._running = False
        # Camera is released by run() on exit to avoid racing cap.read().
