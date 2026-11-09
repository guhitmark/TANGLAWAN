from fastapi import APIRouter
from pydantic import BaseModel
from typing import List
import os
import copy
import requests

class TransformPayload(BaseModel):
    eye: str
    axis: str = None
    rotate: bool = False

class CropPayload(BaseModel):
    eye: str
    points: List[List[float]]

class PatientIdPayload(BaseModel):
    patient_id: str

class DiagnosisPayload(BaseModel):
    diagnosis: str

class SaveMp4Payload(BaseModel):
    save_mp4: bool

class ReconnectPayload(BaseModel):
    left_index: int
    right_index: int

class EnabledPayload(BaseModel):
    left_enabled: bool
    right_enabled: bool

# --- NEW Payload Model for Supabase Sync ---
class SyncSupabasePayload(BaseModel):
    patient_id: str
    dr_name: str
    dr_license: str
    diagnosis: str
    confidence: str
    timestamp: str
    report_html_b64: str

def build_router(state, workers):
    router = APIRouter()

    # --- NEW Route for Direct Supabase Sync ---
    @router.post("/sync_supabase")
    def sync_supabase(payload: SyncSupabasePayload):
        # Your specific Supabase Configuration
        SUPABASE_URL = "https://zkmtfpeoihcxamtjatgh.supabase.co/rest/v1/tanglawan_reports"
        SUPABASE_KEY = "sb_publishable_mIHOi8dllJ0BSou4PY1CZA_XjXCaYgG"
        
        headers = {
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
            "Content-Type": "application/json",
            "Prefer": "return=minimal"
        }
        
        # Structure the row exactly as your Supabase table columns
        data = {
            "patient_id": payload.patient_id,
            "dr_name": payload.dr_name,
            "dr_license": payload.dr_license,
            "diagnosis": payload.diagnosis,
            "confidence": payload.confidence,
            "timestamp": payload.timestamp,
            "report_html_b64": payload.report_html_b64
        }
        
        try:
            response = requests.post(
                SUPABASE_URL, 
                json=data,
                headers=headers,
                timeout=10
            )
            
            if response.status_code in [200, 201]:
                return {"ok": True}
            else:
                return {"ok": False, "error": f"Supabase returned {response.status_code}: {response.text}"}
        except Exception as e:
            return {"ok": False, "error": f"Could not connect to Supabase: {str(e)}"}

    @router.post("/transform")
    def apply_transform(payload: TransformPayload):
        t = state.transform[payload.eye]
        
        if payload.rotate:
            t["rotate"] = (t["rotate"] + 1) % 4
        if payload.axis == "H":
            t["flip_h"] = not t["flip_h"]
        if payload.axis == "V":
            t["flip_v"] = not t["flip_v"]
            
        return {"ok": True}

    @router.post("/crop")
    def apply_crop(payload: CropPayload):
        state.crop_points[payload.eye] = payload.points
        return {"ok": True}

    @router.post("/reset_crop")
    def reset_crop():
        state.reset_crop()
        return {"ok": True, "crop_points": state.crop_points}

    @router.post("/sync_crop/{source_eye}")
    def sync_crop(source_eye: str):
        state.sync_crop(source_eye)
        return {"ok": True, "crop_points": state.crop_points}

    @router.post("/patient_id")
    def set_patient_id(payload: PatientIdPayload):
        state.patient_id = payload.patient_id
        return {"ok": True}

    @router.post("/diagnosis")
    def set_diagnosis(payload: DiagnosisPayload):
        state.diagnosis = payload.diagnosis
        return {"ok": True}

    @router.post("/save_mp4")
    def set_save_mp4(payload: SaveMp4Payload):
        state.save_mp4 = payload.save_mp4
        return {"ok": True}

    @router.post("/reconnect_cameras")
    def reconnect_cameras(payload: ReconnectPayload):
        workers["L"].reconnect(payload.left_index)
        workers["R"].reconnect(payload.right_index)
        return {"ok": True}

    @router.post("/enabled")
    def set_enabled(payload: EnabledPayload):
        state.left_enabled = payload.left_enabled
        state.right_enabled = payload.right_enabled
        return {"ok": True}

    @router.post("/start")
    def start_rec():
        if not state.patient_id:
            return {"ok": False, "error": "Patient ID is required to start recording."}
        state.start_recording()
        return {"ok": True}

    @router.post("/pause")
    def pause_rec():
        state.pause_recording()
        return {"ok": True}

    @router.post("/stop")
    def stop_rec():
        state.stop_recording()
        return {"ok": True}

    @router.post("/discard")
    def discard_rec():
        state.discard_recording()
        return {"ok": True}

    @router.post("/save")
    def save_session():
        base_dir = os.path.expanduser("~/Desktop/vision_capture_v1/03_TANGLAWAN_DATASET")
        left_dir = os.path.join(base_dir, "left")
        right_dir = os.path.join(base_dir, "right")
        
        os.makedirs(left_dir, exist_ok=True)
        os.makedirs(right_dir, exist_ok=True)
        
        success = state.save(left_path=left_dir, right_path=right_dir)
        
        if success:
            return {"ok": True}
        return {"ok": False, "error": "No staged data to save"}
        
    return router