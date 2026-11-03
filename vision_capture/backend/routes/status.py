"""
routes/status.py

"""

from fastapi import APIRouter
import time
import math
from .. import metrics

def build_router(state):
    router = APIRouter()

    def sanitize(val):
        return None if math.isnan(val) else val

    @router.get("/status")
    def get_status():
        state.last_client_ping = time.time()
        
        if state.is_recording and state.elapsed() >= 300:
            state.stop_recording()

        elapsed_s = state.elapsed()
        
        if state.is_recording or state.has_staged_data:
            phase = state.evaluate_phase(elapsed_s)
        else:
            phase = "SYS.STANDBY"

        with state.lock:
            data_l = list(state.session_data["L"])
            data_r = list(state.session_data["R"])

        r_met_L, b_met_L = metrics.get_clinical_report(data_l)
        r_met_R, b_met_R = metrics.get_clinical_report(data_r)

        return {
            "is_recording": state.is_recording,
            "is_paused": state.is_paused,
            "has_staged_data": state.has_staged_data,
            "elapsed": elapsed_s,
            "phase": phase,
            "patient_id": state.patient_id,
            "diagnosis": state.diagnosis,
            "live": {
                "L": {
                    "diameter_mm": sanitize(state.live_diameter_mm["L"]), 
                    "confidence": state.live_confidence["L"], 
                    "fps": state.live_fps["L"]
                },
                "R": {
                    "diameter_mm": sanitize(state.live_diameter_mm["R"]), 
                    "confidence": state.live_confidence["R"], 
                    "fps": state.live_fps["R"]
                },
            },
            "clinical": {
                "L": {"red": r_met_L, "blue": b_met_L},
                "R": {"red": r_met_R, "blue": b_met_R},
            },
        }
    return router