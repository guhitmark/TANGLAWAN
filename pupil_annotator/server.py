import os, cv2, numpy as np, json, shutil, random
from fastapi import APIRouter, Request
from fastapi.responses import FileResponse

annotator_router = APIRouter()

TRACKER_ROOT = "/home/tanglawan/Desktop/vision_capture_v1/dataset/PUPIL_TRACKER"
ANNOTATOR_WEB = "/home/tanglawan/Desktop/vision_capture_v1/pupil_annotator/web"
EXPORT_ROOT = "/home/tanglawan/Desktop/vision_capture_v1/pupil_annotator/annotated"
LOGO_PATH = "/home/tanglawan/Desktop/vision_capture_v1/vision_capture/web/assets/logo.png"

os.makedirs(TRACKER_ROOT, exist_ok=True)
os.makedirs(EXPORT_ROOT, exist_ok=True)

@annotator_router.get("/annotator")
def serve_annotator_ui():
    return FileResponse(os.path.join(ANNOTATOR_WEB, "annotator.html"))

@annotator_router.get("/web/annotator.js")
def serve_annotator_js():
    return FileResponse(os.path.join(ANNOTATOR_WEB, "annotator.js"))

@annotator_router.get("/logo.png")
def serve_logo():
    return FileResponse(LOGO_PATH)

@annotator_router.get("/api/annotator/dataset/{mother}/{base_name}/frames/{frame}")
def serve_frame(mother: str, base_name: str, frame: str):
    path = os.path.join(TRACKER_ROOT, mother, base_name, "frames", frame)
    if os.path.exists(path):
        return FileResponse(path)
    return {"ok": False, "error": "Frame not found"}

@annotator_router.get("/api/annotator/folders")
def get_folders():
    folders_data = []
    if os.path.exists(TRACKER_ROOT):
        for mother in sorted(os.listdir(TRACKER_ROOT)):
            mother_path = os.path.join(TRACKER_ROOT, mother)
            if not os.path.isdir(mother_path):
                continue
                
            avis = sorted([f for f in os.listdir(mother_path) if f.lower().endswith((".avi", ".mp4", ".mkv"))])
            if not avis:
                continue

            trials = []
            for avi in avis:
                base_name = os.path.splitext(avi)[0] 
                frames_dir = os.path.join(mother_path, base_name, "frames")
                
                frames = []
                labeled = []
                
                if os.path.exists(frames_dir):
                    frames = sorted([img for img in os.listdir(frames_dir) if img.endswith(".jpg")])
                    labeled = [img for img in frames if os.path.exists(os.path.join(frames_dir, img.replace(".jpg", "_label.json")))]
                
                trials.append({
                    "id": f"{mother}/{base_name}", 
                    "name": base_name,
                    "frames": frames, 
                    "labeled": labeled
                })
            
            # Send a nested tree structure
            folders_data.append({
                "mother_name": mother,
                "trials": trials
            })
                    
    return {"ok": True, "folders": folders_data}

@annotator_router.post("/api/annotator/extract")
async def extract_api(req: Request):
    data = await req.json()
    folder_path_req = data.get("trial_folder") 
    target_frames = int(data.get("frames", 100))
    
    if "/" not in folder_path_req:
        return {"ok": False, "error": "Invalid folder structure."}
        
    mother, base_name = folder_path_req.split("/", 1)
    mother_path = os.path.join(TRACKER_ROOT, mother)
    
    video_path = None
    for ext in [".avi", ".mp4", ".mkv"]:
        temp_path = os.path.join(mother_path, f"{base_name}{ext}")
        if os.path.exists(temp_path):
            video_path = temp_path
            break
            
    if not video_path:
        return {"ok": False, "error": f"No video file found for '{base_name}'."}
        
    output_dir = os.path.join(mother_path, base_name, "frames")
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    actual_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if actual_frames == 0:
        cap.release()
        return {"ok": False, "error": "Video is empty."}
        
    limit = min(1800, actual_frames)
    step = limit / float(target_frames)
    extracted_total = 0
    
    for i in range(target_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i * step))
        ret, frame = cap.read()
        if ret:
            out_name = f"{base_name}_{i+1:03d}.jpg"
            cv2.imwrite(os.path.join(output_dir, out_name), frame)
            extracted_total += 1
            
    cap.release()
    return {"ok": True, "extracted": extracted_total}

@annotator_router.post("/api/annotator/save_annotation")
async def save_annotation(req: Request):
    data = await req.json()
    folder_path_req = data.get("trial")
    mother, base_name = folder_path_req.split("/", 1)
    frame_names = data.get("frames", [])
    ellipse = data.get("ellipse")
    
    frames_dir = os.path.join(TRACKER_ROOT, mother, base_name, "frames")
    
    is_empty = not ellipse or ellipse.get('rx', 0) == 0 or ellipse.get('ry', 0) == 0
    
    for frame_name in frame_names:
        img_base = frame_name.replace(".jpg", "")
        json_path = os.path.join(frames_dir, f"{img_base}_label.json")
        mask_path = os.path.join(frames_dir, f"{img_base}_mask.png")
        
        if is_empty:
            if os.path.exists(json_path): os.remove(json_path)
            if os.path.exists(mask_path): os.remove(mask_path)
        else:
            with open(json_path, "w") as f:
                json.dump(ellipse, f)
                
            mask = np.zeros((400, 400), dtype=np.uint8)
            center = (int(ellipse['cx']), int(ellipse['cy']))
            axes = (int(ellipse['rx']), int(ellipse['ry']))
            angle = ellipse['angle']
            cv2.ellipse(mask, center, axes, angle, 0, 360, 255, -1)
            cv2.imwrite(mask_path, mask)
            
    return {"ok": True, "status": "deleted" if is_empty else "saved"}

@annotator_router.post("/api/annotator/export")
async def export_annotated(req: Request):
    labeled_files = []
    
    # Globally scan all mothers and trials to unify the entire annotated dataset
    if not os.path.exists(TRACKER_ROOT):
        return {"ok": False, "error": "Dataset tracking root not found."}
        
    for mother in os.listdir(TRACKER_ROOT):
        mother_path = os.path.join(TRACKER_ROOT, mother)
        if not os.path.isdir(mother_path): continue
        for base_name in os.listdir(mother_path):
            src_dir = os.path.join(mother_path, base_name, "frames")
            if not os.path.exists(src_dir): continue
            
            for f in os.listdir(src_dir):
                if f.endswith("_label.json"):
                    base_file = f.replace("_label.json", "")
                    labeled_files.append((src_dir, base_file))
                    
    if not labeled_files:
        return {"ok": False, "error": "No annotations found across any folders."}

    # Global dataset shuffle and split
    random.shuffle(labeled_files)
    n = len(labeled_files)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    
    splits = {
        "train": labeled_files[:n_train],
        "val": labeled_files[n_train:n_train+n_val],
        "test": labeled_files[n_train+n_val:]
    }
    
    def copy_split(split_name, items):
        img_dir = os.path.join(EXPORT_ROOT, split_name, "images")
        mask_dir = os.path.join(EXPORT_ROOT, split_name, "masks")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)
        
        for src_dir, base in items:
            img_src = os.path.join(src_dir, f"{base}.jpg")
            mask_src = os.path.join(src_dir, f"{base}_mask.png")
            
            if os.path.exists(img_src) and os.path.exists(mask_src):
                # Using base prevents overlap since your extracted frames are already named 'base_name_001.jpg'
                shutil.copy2(img_src, os.path.join(img_dir, f"{base}.jpg"))
                shutil.copy2(mask_src, os.path.join(mask_dir, f"{base}_mask.png"))

    copy_split("train", splits["train"])
    copy_split("val", splits["val"])
    copy_split("test", splits["test"])
            
    return {"ok": True, "count": len(labeled_files)}

@annotator_router.post("/api/annotator/discard")
async def discard_extracted(req: Request):
    data = await req.json()
    folder_path_req = data.get("trial_folder")
    if not folder_path_req: return {"ok": False, "error": "No folder selected."}
    
    mother, base_name = folder_path_req.split("/", 1)
    frames_dir = os.path.join(TRACKER_ROOT, mother, base_name, "frames")
    
    if os.path.exists(frames_dir): 
        shutil.rmtree(frames_dir)
    return {"ok": True}

@annotator_router.post("/api/annotator/delete_frames")
async def delete_frames(req: Request):
    data = await req.json()
    folder_path_req = data.get("trial")
    frame_names = data.get("frames", [])
    
    if not folder_path_req or not frame_names:
        return {"ok": False, "error": "Invalid request parameters."}
        
    mother, base_name = folder_path_req.split("/", 1)
    frames_dir = os.path.join(TRACKER_ROOT, mother, base_name, "frames")
    
    deleted_count = 0
    for frame_name in frame_names:
        img_base = frame_name.replace(".jpg", "")
        
        # Paths to all possible related files
        jpg_path = os.path.join(frames_dir, f"{img_base}.jpg")
        json_path = os.path.join(frames_dir, f"{img_base}_label.json")
        mask_path = os.path.join(frames_dir, f"{img_base}_mask.png")
        
        try:
            if os.path.exists(jpg_path):
                os.remove(jpg_path)
                deleted_count += 1
            if os.path.exists(json_path):
                os.remove(json_path)
            if os.path.exists(mask_path):
                os.remove(mask_path)
        except Exception as e:
            return {"ok": False, "error": str(e)}
            
    return {"ok": True, "deleted": deleted_count}

@annotator_router.get("/api/annotator/dataset/{mother}/{base_name}/label/{frame}")
def get_frame_label(mother: str, base_name: str, frame: str):
    label_filename = frame.replace(".jpg", "_label.json")
    path = os.path.join(TRACKER_ROOT, mother, base_name, "frames", label_filename)
    if os.path.exists(path):
        with open(path, "r") as f:
            data = json.load(f)
        return {"ok": True, "ellipse": data}
    return {"ok": True, "ellipse": None}