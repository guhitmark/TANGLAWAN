"""
main.py
"""

import os
import time
import threading
import cv2
import psutil
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from .session_state import SessionState
from .camera_worker import CameraWorker
from .stimulus_led import StimulusController
from .gpio_manager import init_gpio  # <-- ADDED INITIALIZER
from .battery import read_battery_status
from .routes import commands, stream, status

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEB_DIR  = os.path.join(ROOT_DIR, "web")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/css",    StaticFiles(directory=os.path.join(WEB_DIR, "css")),    name="css")
app.mount("/js",     StaticFiles(directory=os.path.join(WEB_DIR, "js")),     name="js")
app.mount("/assets", StaticFiles(directory=os.path.join(WEB_DIR, "assets")), name="assets")
app.mount("/dataset", StaticFiles(directory="/home/tanglawan/Desktop/vision_capture_v1/dataset"), name="dataset")

state   = SessionState(root_dir=ROOT_DIR)
workers = {}
stimulus_controller  = None

export_lock          = threading.Lock()

@app.on_event("startup")
def startup():
    global stimulus_controller

    # FIXED: Actually initialize the board before the threads start
    init_gpio()

    active_ports = [i for i in range(10) if os.path.exists(f"/dev/video{i}")]
    port_L = active_ports[0] if len(active_ports) > 0 else 0
    port_R = active_ports[1] if len(active_ports) > 1 else 1

    def _apply_v4l2_locks(index):
        # Sweet-spot exposure (95 = 9.5ms) and gain (1) to prevent glare washout
        os.system(f"v4l2-ctl -d /dev/video{index} -c exposure_auto=1 2>/dev/null")
        os.system(f"v4l2-ctl -d /dev/video{index} -c auto_exposure=1 2>/dev/null")
        os.system(f"v4l2-ctl -d /dev/video{index} -c exposure_auto_priority=0 2>/dev/null")
        os.system(f"v4l2-ctl -d /dev/video{index} -c exposure_absolute=5 2>/dev/null")
        os.system(f"v4l2-ctl -d /dev/video{index} -c exposure_time_absolute=15 2>/dev/null")
        os.system(f"v4l2-ctl -d /dev/video{index} -c gain=1 2>/dev/null")

    def _open(index):
        _apply_v4l2_locks(index)

        # 1. Negotiate 120/1 with OV9281 hardware, throttle to 30/1 BEFORE decoding
        pipeline = (
            f"v4l2src device=/dev/video{index} "
            f"! image/jpeg,width=640,height=400,framerate=120/1 "
            f"! videorate "
            f"! image/jpeg,framerate=30/1 "
            f"! jpegdec "
            f"! videoconvert "
            f"! video/x-raw,format=BGR "
            f"! appsink drop=1 max-buffers=1"
        )
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

        # 2. Fallback to standard 120/1 GStreamer pipeline with appsink frame dropping
        if not cap.isOpened():
            pipeline_fallback = (
                f"v4l2src device=/dev/video{index} "
                f"! image/jpeg,width=640,height=400,framerate=120/1 "
                f"! jpegdec "
                f"! videoconvert "
                f"! video/x-raw,format=BGR "
                f"! appsink drop=1 max-buffers=1"
            )
            cap = cv2.VideoCapture(pipeline_fallback, cv2.CAP_GSTREAMER)
        
        # 3. Fallback to standard 1-argument V4L2 capture (fixes OpenCV 3.x TypeError)
        if not cap.isOpened():
            cap = cv2.VideoCapture(index)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))  
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 400)   
                cap.set(cv2.CAP_PROP_FPS, 30.0)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Re-apply V4L2 locks AFTER opening so OpenCV's internal init doesn't reset them
        _apply_v4l2_locks(index)
        return cap

    cap_L = _open(port_L)
    cap_R = _open(port_R)

    if not cap_R.isOpened():
        cap_R.release()
        cap_R = _open(port_L)

    workers["L"] = CameraWorker("L", port_L, state, cap=cap_L)
    workers["R"] = CameraWorker("R", port_R, state, cap=cap_R)
    workers["L"].start()
    workers["R"].start()

    stimulus_controller = StimulusController(state)
    stimulus_controller.start()

    # Browser launch is handled by run.py, which polls until the server
    # is actually ready (after the AI engine finishes loading).

@app.on_event("shutdown")
def shutdown():
    for w in workers.values():
        w.stop()
    if stimulus_controller: stimulus_controller.stop()

@app.get("/")
def serve_index():
    return FileResponse(os.path.join(WEB_DIR, "index.html"))

@app.get("/recorder")
def serve_recorder():
    return FileResponse(os.path.join(WEB_DIR, "recorder.html"))

@app.get("/api/hardware_status")
def hardware_status():
    is_jetson = (os.path.exists("/etc/nv_tegra_release") or
                 os.path.exists("/usr/lib/aarch64-linux-gnu/tegra"))
    if not is_jetson:
        try:
            import Jetson.GPIO  # noqa: F401
            is_jetson = True
        except ImportError:
            pass
    cams = sum(1 for i in range(6) if os.path.exists(f"/dev/video{i}"))
    return {"jetson_connected": is_jetson, "cameras_detected": cams}

@app.get("/api/sys_resources")
def get_sys_resources():
    fan_pwm = 0
    try:
        with open("/sys/devices/pwm-fan/target_pwm", "r") as f:
            fan_pwm = int(f.read().strip())
    except Exception:
        pass

    temp_c = 0.0
    try:
        with open("/sys/devices/virtual/thermal/thermal_zone0/temp", "r") as f:
            temp_c = float(f.read().strip()) / 1000.0
    except Exception:
        pass
    
    return {
        "ok": True,
        "fan_pwm": fan_pwm,
        "ram_percent": psutil.virtual_memory().percent,
        "disk_percent": psutil.disk_usage('/').percent,
        "temp_c": temp_c
    }

@app.get("/api/battery")
def get_battery():
    batt = read_battery_status()
    return {
        "percent":    f"{batt['percentage']}%",
        "percentage": batt['percentage'],
        "voltage":    batt['voltage'],
        "charging":   batt['charging'],
        "ok":         batt['ok'],
    }

@app.post("/api/system/shutdown")
def system_shutdown():
    def _execute_shutdown():
        try:
            with open("/sys/devices/pwm-fan/target_pwm", "w") as f:
                f.write("0")
        except Exception:
            pass
            
        time.sleep(1.5)
        os.system("sudo shutdown now")
        
    threading.Thread(target=_execute_shutdown, daemon=True).start()
    return {"ok": True, "message": "Initiating safe hardware shutdown"}

@app.post("/api/recorder/ping")
def ping_heartbeat():
    state.last_client_ping = time.time()
    return {"ok": True}

# ---> NEW ENDPOINT FOR LAB VIEW IR TOGGLE <---
@app.post("/api/recorder/lab_view")
async def toggle_lab_view(request: Request):
    try:
        data = await request.json()
        state.lab_view_active = data.get("active", False)
    except:
        pass
    return {"ok": True}

@app.post("/api/recorder/save_trial")
async def save_trial(request: Request):
    try:
        payload = await request.json()
    except:
        payload = {}

    with export_lock:
        if not state.has_staged_data:
            return {"ok": True, "saved": ["Data already successfully compiled."]}
        try:
            state.stop_recording()
            
            response_data = {}
            
            if getattr(state, "diagnosis", "") == "UNKNOWN":
                from . import inference
                inference_result = inference.run_inference(state, payload)

                raw_path = inference_result.get("report_path", "")
                web_url = raw_path.replace("/home/tanglawan/Desktop/vision_capture_v1/dataset", "/dataset")
                inference_result["report_url"] = web_url
                
                response_data["inference"] = inference_result
                
            from . import exporters
            result = exporters.export_all(state)
            
            if result.get("ok"):
                response_data["ok"] = True
                response_data["saved"] = result.get("status")
                return response_data
            else:
                return {"ok": False, "error": result.get("error")}
        except Exception as e:
            return {"ok": False, "error": str(e)}

@app.post("/api/recorder/new_trial")
def new_trial():
    state.discard_recording()
    with state.lock:
        # THE FIX: Do not erase the patient_id if we are recording a dataset!
        if not getattr(state, "tracker_mode", False):
            state.patient_id   = ""
            state.diagnosis    = "Select Diagnosis..."
            
        state.session_data = {"L": [], "R": []}
    return {"ok": True}

app.include_router(commands.build_router(state, workers), prefix="/api/recorder")
app.include_router(stream.build_router(workers),          prefix="/api/recorder")
app.include_router(status.build_router(state),            prefix="/api/recorder")

import sys
sys.path.append("/home/tanglawan/Desktop/vision_capture_v1/pupil_annotator")
from server import annotator_router
app.include_router(annotator_router)