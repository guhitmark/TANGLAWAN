from fastapi import APIRouter
from pydantic import BaseModel
from typing import List
import os
import copy

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

def build_router(state, workers):
    router = APIRouter()

    @router.post("/transform")
    def apply_transform(payload: TransformPayload):
        # Update the transform dictionary in the central state, not on the worker thread
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
        # Hardcoded static paths to enforce dataset integrity
        base_dir = os.path.expanduser("~/Desktop/vision_capture_v1/03_TANGLAWAN_DATASET")
        left_dir = os.path.join(base_dir, "left")
        right_dir = os.path.join(base_dir, "right")
        
        # Ensure target directories exist before writing
        os.makedirs(left_dir, exist_ok=True)
        os.makedirs(right_dir, exist_ok=True)
        
        # Execute save command explicitly mapping to verified eye directories
        success = state.save(left_path=left_dir, right_path=right_dir)
        
        if success:
            return {"ok": True}
        return {"ok": False, "error": "No staged data to save"}
        
    return router