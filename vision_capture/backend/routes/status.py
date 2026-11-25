
"""
routes/status.py
TANGLAWAN — 90-second chromatic pupillometry protocol
"""

from fastapi import APIRouter
import time
import math
from .. import metrics

CLINICAL_DURATION_S = 90
TRACKER_DURATION_S = 60


def build_router(state):
    router = APIRouter()

    def sanitize(val):
        try:
            x = float(val)
            return x if math.isfinite(x) else None
        except (TypeError, ValueError):
            return None

    @router.get("/status")
    def get_status():
        state.last_client_ping = time.time()

        target = (
            TRACKER_DURATION_S
            if getattr(state, "tracker_mode", False)
            else CLINICAL_DURATION_S
        )

        # Automatically stop at 90 seconds (60 for tracker mode).
        # Do not discard staged data.
        if state.is_recording and state.elapsed() >= target:
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
            "protocol_duration": target,
            "protocol_complete": bool(
                state.has_staged_data
                and not state.is_recording
                and elapsed_s >= target
            ),
            "phase": phase,
            "patient_id": state.patient_id,
            "diagnosis": state.diagnosis,
            "live": {
                "L": {
                    "diameter_mm": sanitize(
                        state.live_diameter_mm["L"]
                    ),
                    "confidence": state.live_confidence["L"],
                    "fps": state.live_fps["L"]
                },
                "R": {
                    "diameter_mm": sanitize(
                        state.live_diameter_mm["R"]
                    ),
                    "confidence": state.live_confidence["R"],
                    "fps": state.live_fps["R"]
                },
            },
            "clinical": {
                "L": {
                    "red": r_met_L,
                    "blue": b_met_L
                },
                "R": {
                    "red": r_met_R,
                    "blue": b_met_R
                },
            },
        }

    return router
