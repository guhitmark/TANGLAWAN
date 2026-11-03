import os
from datetime import datetime

class TrackerModeOverride:
    """
    MASTER OVERRIDE FILE for RITnet Tracker Mode.
    Isolates all dataset bypass logic (pathing, exports, telemetry blocks) away from the clinical core.
    """
    @staticmethod
    def is_active(state):
        return getattr(state, "diagnosis", "") == "RITNET_TRACKER"

    @staticmethod
    def get_paths(state):
        """Generates exact <filename>/<filename>_[timestamp]_L.avi routing paths."""
        filename = state.patient_id.strip()
        
        if len(filename) == 0:
            return None, None, False, "Input Filename cannot be empty in Dataset Mode"
        
        # Generate a unique timestamp for this specific trial
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Locks directory specifically to dataset/PUPIL_TRACKER/<filename>
        base_dir = f"/home/tanglawan/Desktop/vision_capture_v1/dataset/PUPIL_TRACKER/{filename}"
        os.makedirs(base_dir, exist_ok=True)
        
        # Applies eye#/eye#_[timestamp]_L.avi naming convention to prevent overwriting
        path_L = os.path.join(base_dir, f"{filename}_{timestamp}_L.avi")
        path_R = os.path.join(base_dir, f"{filename}_{timestamp}_R.avi")
        
        return path_L, path_R, True, ""

    @staticmethod
    def finalize_dataset(state):
        """
        Bypasses the entire clinical exporter engine (Plots, CSVs, Excel, Inference).
        The raw videos were already written directly to the target folder by the camera worker.
        """
        filename = state.patient_id.strip()
        
        with state.lock:
            state.has_staged_data = False
            state.staged_left     = False
            state.staged_right    = False
            state.patient_id      = ""
            state.diagnosis       = "Select Diagnosis..."
            
        return {
            "ok": True, 
            "error": None, 
            "status": [
                "RAW DATASET SAVED", 
                "Tracking Overlays Bypassed", 
                "Clinical Telemetry Bypassed"
            ], 
            "trial_folder": filename
        }