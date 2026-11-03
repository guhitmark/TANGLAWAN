import os
import threading
import time

from controllers.pupil_pipeline import PupilPipeline
from controllers.tracker_config import TARGET_FPS
from .mode_manager import TrackerModeOverride

# UPDATED: Total clinical protocol time changed from 300 to 180 seconds (3 minutes)
PROTOCOL_TOTAL_SECONDS = 180
# UPDATED: Red window remains 0-60. Blue window shifts to 120-180.
RED_WINDOW = (0, 60)
BLUE_WINDOW = (120, 180)

DIAGNOSIS_OPTIONS = [
    "Select Diagnosis...",
    "LBL_0 (No DR)",
    "LBL_1 (Diabetic-DR)",
    "LBL_2 (Diabetic-NoDR)",
    "UNKNOWN",
    "RITNET_TRACKER"
]

class SessionState:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.lock = threading.RLock()
        self.pupil = PupilPipeline()

        self.is_recording = False
        self.is_paused = False
        self.has_staged_data = False
        self.start_time = 0.0
        self.frozen_elapsed = 0  
        self.lab_view_active = False 
        
        self.tracker_mode = False 
        
        self.writer_L = None
        self.writer_R = None
        self.last_client_ping = time.time() 

        self.patient_id = ""
        self.diagnosis = DIAGNOSIS_OPTIONS[0]
        self.left_enabled = True
        self.right_enabled = True
        self.save_mp4 = True

        self.transform = {
            "L": {"flip_h": False, "flip_v": False, "rotate": 1},
            "R": {"flip_h": False, "flip_v": False, "rotate": 0},
        }
        self.CROP_W = 400
        self.CROP_H = 400

        default_pts = [[40, 40], [440, 40], [440, 440], [40, 440]]
        self.crop_points = {
            "L": [p[:] for p in default_pts],
            "R": [p[:] for p in default_pts],
        }

        self.staged_left = False
        self.staged_right = False
        self.stage_path_L = os.path.join(root_dir, ".temp_stage_L.avi")
        self.stage_path_R = os.path.join(root_dir, ".temp_stage_R.avi")
        self.frames_captured_L = 0
        self.frames_captured_R = 0

        self.session_data = {"L": [], "R": []}
        self.live_diameter_mm = {"L": float("nan"), "R": float("nan")}
        self.live_confidence = {"L": 0.0, "R": 0.0}
        self.live_fps = {"L": TARGET_FPS, "R": TARGET_FPS}

    def _default_crop_points(self):
        return [[40, 40], [440, 40], [440, 440], [40, 440]]

    def elapsed(self):
        if self.is_recording:
            max_time = 60 if self.tracker_mode else PROTOCOL_TOTAL_SECONDS
            return min(int(time.time() - self.start_time), max_time)
        if self.has_staged_data:
            return self.frozen_elapsed  
        return 0

    def is_active_window(self, elapsed_s):
        if self.tracker_mode: return True
        return elapsed_s <= RED_WINDOW[1] or BLUE_WINDOW[0] <= elapsed_s <= BLUE_WINDOW[1]

    def evaluate_phase(self, elapsed_s):
        if self.tracker_mode:
            return "DATASET RECORDING [60s]"
            
        if elapsed_s < 10:
            return "BASELINE [RED]"
        elif elapsed_s < 11:
            return "STIMULUS: RED PULSE"
        elif elapsed_s < 60:
            return "PIPR [RED]"
        # UPDATED: Dark adaptation now correctly spans exactly 60 seconds (60s to 120s)
        elif elapsed_s < 120:
            return "DARK ADAPTATION"
        # UPDATED: Blue sequence begins at 120s
        elif elapsed_s < 130:
            return "BASELINE [BLUE]"
        elif elapsed_s < 131:
            return "STIMULUS: BLUE PULSE"
        elif elapsed_s < 180:
            return "PIPR [BLUE]"
        else:
            return "SEQUENCE COMPLETE"

    def start_recording(self):
        import cv2
        with self.lock:
            if self.is_recording:
                return {"ok": False, "error": "Already recording"}
            
            # Use Master Override to define state
            self.tracker_mode = TrackerModeOverride.is_active(self)

            if self.tracker_mode:
                path_L, path_R, valid, err_msg = TrackerModeOverride.get_paths(self)
                if not valid:
                    return {"ok": False, "error": err_msg}
                self.stage_path_L = path_L
                self.stage_path_R = path_R
            else:
                if len(self.patient_id) != 6 or not self.patient_id.isdigit():
                    return {"ok": False, "error": "Patient ID must be exactly 6 digits"}
                
                self.stage_path_L = os.path.join(self.root_dir, ".temp_stage_L.avi")
                self.stage_path_R = os.path.join(self.root_dir, ".temp_stage_R.avi")
                
            if not self.left_enabled and not self.right_enabled:
                return {"ok": False, "error": "Enable at least one eye (Left or Right)"}

            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            dim = (self.CROP_W, self.CROP_H)
            self.writer_L = cv2.VideoWriter(self.stage_path_L, fourcc, TARGET_FPS, dim) if self.left_enabled else None
            self.writer_R = cv2.VideoWriter(self.stage_path_R, fourcc, TARGET_FPS, dim) if self.right_enabled else None

            self.staged_left = self.left_enabled
            self.staged_right = self.right_enabled
            self.frames_captured_L = 0
            self.frames_captured_R = 0
            self.session_data = {"L": [], "R": []}

            self.is_recording = True
            self.is_paused = False
            self.has_staged_data = True
            self.frozen_elapsed = 0
            self.start_time = time.time()

            self.pupil.start_session()
            return {"ok": True}

    def pause_recording(self):
        with self.lock:
            if not self.is_recording:
                return {"ok": False, "error": "Not recording"}
            self.is_paused = not self.is_paused
            return {"ok": True, "is_paused": self.is_paused}

    def stop_recording(self):
        with self.lock:
            if not self.is_recording:
                return {"ok": False, "error": "Not recording"}
            
            self.frozen_elapsed = self.elapsed()
            
            # FIX: Only flip the flags. Do NOT run writer.release() here.
            # We must delegate the release command to the camera_worker thread 
            # to guarantee the SD card index is written cleanly.
            self.is_recording = False
            self.is_paused = False
            
            # Sleep briefly to ensure the camera loop detects the flag drop
            time.sleep(0.5) 
            
            return {"ok": True}

    def discard_recording(self):
        with self.lock:
            # 1. Stop the active recording cleanly
            if self.is_recording:
                self.stop_recording()
            
            # 2. Only delete files if we are discarding an unsaved trial.
            # If save_trial was already executed, has_staged_data is False, so this safely skips.
            if self.has_staged_data:
                for p in (self.stage_path_L, self.stage_path_R):
                    if p and os.path.exists(p):
                        try: os.remove(p)
                        except Exception: pass
                            
            # 3. UNIVERSAL RESET: Wipe all internal flags and UI state variables
            self.has_staged_data = False
            self.staged_left = False
            self.staged_right = False
            self.frozen_elapsed = 0
            
            self.patient_id = ""
            self.diagnosis = DIAGNOSIS_OPTIONS[0]
            self.tracker_mode = False 
            
            self.stage_path_L = os.path.join(self.root_dir, ".temp_stage_L.avi")
            self.stage_path_R = os.path.join(self.root_dir, ".temp_stage_R.avi")
            
            return {"ok": True}

    def reset_crop(self):
        with self.lock:
            default_pts = self._default_crop_points()
            self.crop_points["L"] = default_pts.copy()
            self.crop_points["R"] = [p[:] for p in default_pts]
            return {"ok": True}

    def sync_crop(self, source_eye):
        with self.lock:
            dest_eye = "R" if source_eye == "L" else "L"
            pts = self.crop_points[source_eye]
            self.crop_points[dest_eye] = [p[:] for p in pts]
            return {"ok": True}