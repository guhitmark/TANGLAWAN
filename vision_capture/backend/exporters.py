import os
import json
from datetime import datetime

import numpy as np
import cv2

from . import metrics
from .mode_manager import TrackerModeOverride

# ── Fixed dataset root ────────────────────────────────────────────
DATASET_ROOT = "/home/tanglawan/Desktop/vision_capture_v1/dataset"


def _finalize_videos(temp_avi, final_avi, final_mp4, actual_fps, save_mp4):
    if not os.path.exists(temp_avi):
        return
    cap = cv2.VideoCapture(temp_avi)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_avi = cv2.VideoWriter(final_avi, cv2.VideoWriter_fourcc(*"MJPG"), actual_fps, (w, h))
    out_mp4 = None
    if save_mp4:
        out_mp4 = cv2.VideoWriter(final_mp4, cv2.VideoWriter_fourcc(*"mp4v"), actual_fps, (w, h))

    try:
        while True:
            ret, frame = cap.read()
            if not ret: break
            out_avi.write(frame)
            if out_mp4: out_mp4.write(frame)
    finally:
        cap.release()
        out_avi.release()
        if out_mp4: out_mp4.release()
        
        if os.path.exists(temp_avi):
            try: os.remove(temp_avi)
            except Exception: pass


def export_graph_png(session_data_eye, export_path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not session_data_eye:
        raise ValueError("No telemetry data to plot.")

    fig, (ax_red, ax_blue) = plt.subplots(1, 2, figsize=(14, 5), dpi=120)
    fig.suptitle(title, fontsize=13, fontweight="bold")

    # UPDATED TIMELINE FOR THE 3-MINUTE PROTOCOL (120s to 180s)
    for ax, xlim, label, color in [
        (ax_red,  (0,   60),  "Red Stimulus PLR",  "#dc2626"),
        (ax_blue, (120, 180), "Blue Stimulus PLR", "#2563eb"),
    ]:
        ax.set_title(label, color=color, fontweight="bold")
        ax.set_xlim(*xlim)
        ax.set_ylim(1.0, 9.0) 
        ax.set_xlabel("Time (s)", fontsize=9)
        ax.set_ylabel("Pupil Diameter (mm)", fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.tick_params(labelsize=8)

    t_r_line, y_r_line, t_r_drop = [], [], []
    t_b_line, y_b_line, t_b_drop = [], [], []
    
    # --- FORWARD FILL VARIABLES ---
    last_valid_red = np.nan
    last_valid_blue = np.nan
    
    for r in session_data_eye:
        t, valid = r["t"], r["valid"]
        
        try:
            y = float(r["y"]) if r["y"] is not None else np.nan
        except (TypeError, ValueError, KeyError):
            y = np.nan
            
        is_good = valid and not np.isnan(y) and y > 0
        
        if t <= 60:
            if is_good:
                last_valid_red = y
            else:
                t_r_drop.append(t)
            
            if not np.isnan(last_valid_red):
                t_r_line.append(t)
                y_r_line.append(last_valid_red)
                
        # UPDATED TIMELINE FOR THE 3-MINUTE PROTOCOL (120s to 180s)
        elif 120 <= t <= 180:
            if is_good:
                last_valid_blue = y
            else:
                t_b_drop.append(t)
                
            if not np.isnan(last_valid_blue):
                t_b_line.append(t)
                y_b_line.append(last_valid_blue)

    if t_r_line: ax_red.plot(t_r_line, y_r_line, color="#dc2626", lw=1.5, label="Valid Track")
    if t_b_line: ax_blue.plot(t_b_line, y_b_line, color="#2563eb", lw=1.5, label="Valid Track")

    if t_r_drop: ax_red.scatter(t_r_drop, [2.0]*len(t_r_drop), color="#f59e0b", marker="x", s=30, lw=1.5, label="Tracking Lost", zorder=5)
    if t_b_drop: ax_blue.scatter(t_b_drop, [2.0]*len(t_b_drop), color="#f59e0b", marker="x", s=30, lw=1.5, label="Tracking Lost", zorder=5)

    for ax in (ax_red, ax_blue):
        ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(export_path, bbox_inches="tight")
    plt.close(fig)


def export_interactive_html(session_data_eye, export_path, title):
    import plotly.graph_objects as go

    if not session_data_eye:
        raise ValueError("No telemetry data to plot.")

    t_r_line, y_r_line, t_r_drop = [], [], []
    t_b_line, y_b_line, t_b_drop = [], [], []
    
    # --- FORWARD FILL VARIABLES ---
    last_valid_red = np.nan
    last_valid_blue = np.nan
    
    for r in session_data_eye:
        t, valid = r["t"], r["valid"]
        try:
            y = float(r["y"]) if r["y"] is not None else np.nan
        except (TypeError, ValueError, KeyError):
            y = np.nan
            
        is_good = valid and not np.isnan(y) and y > 0
        
        if t <= 60:
            if is_good:
                last_valid_red = y
            else:
                t_r_drop.append(t)
                
            if not np.isnan(last_valid_red):
                t_r_line.append(t)
                y_r_line.append(last_valid_red)
                
        # UPDATED TIMELINE FOR THE 3-MINUTE PROTOCOL (120s to 180s)
        elif 120 <= t <= 180:
            if is_good:
                last_valid_blue = y
            else:
                t_b_drop.append(t)
                
            if not np.isnan(last_valid_blue):
                t_b_line.append(t)
                y_b_line.append(last_valid_blue)

    fig = go.Figure()
    
    if t_r_line: fig.add_trace(go.Scatter(x=t_r_line, y=y_r_line, mode="lines",
        name="Red Stimulus", line=dict(color="#dc2626", width=2)))
        
    if t_b_line: fig.add_trace(go.Scatter(x=t_b_line, y=y_b_line, mode="lines",
        name="Blue Stimulus", line=dict(color="#2563eb", width=2)))
        
    if t_r_drop: fig.add_trace(go.Scatter(x=t_r_drop, y=[2.0]*len(t_r_drop), mode="markers",
        name="Red Tracking Lost", marker=dict(color="#f59e0b", symbol="x", size=6)))
        
    if t_b_drop: fig.add_trace(go.Scatter(x=t_b_drop, y=[2.0]*len(t_b_drop), mode="markers",
        name="Blue Tracking Lost", marker=dict(color="#f59e0b", symbol="x", size=6)))

    fig.update_layout(
        title=f"Interactive PLR — {title}",
        xaxis_title="Time (s)", yaxis_title="Pupil Diameter (mm)",
        template="plotly_dark", hovermode="x unified", dragmode="zoom",
        yaxis=dict(range=[1.0, 9.0]),
        # UPDATED: Changed the X-axis limit to 180s
        xaxis=dict(range=[0, 180], rangeslider=dict(visible=True, thickness=0.08)),
    )
    fig.write_html(export_path, config={"scrollZoom": True, "displayModeBar": True})


def export_excel_report(session_data_eye, export_path, r_met, b_met):
    import pandas as pd

    if not session_data_eye:
        raise ValueError("No telemetry data to export.")

    df = pd.DataFrame(session_data_eye)
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    
    df["Red Stimulus (mm)"]  = np.where(df["t"] <= 60, df["y"], np.nan)
    # UPDATED TIMELINE FOR THE 3-MINUTE PROTOCOL (120s to 180s)
    df["Blue Stimulus (mm)"] = np.where((df["t"] >= 120) & (df["t"] <= 180), df["y"], np.nan)
    df.rename(columns={"t": "Time (s)", "y": "Raw Diameter (mm)", "valid": "Valid Track"},
              inplace=True)

    PARAM_LABELS = [
        "Baseline Diam. [D_base]", "Minimum Diam. [D_min]",
        "Constriction Amplitude",  "Constriction Latency (ms)",
        "Max Constrict. Velocity", "Avg Constrict. Velocity",
        "PIPR (mm)", "Max Redilation Velocity", "75% Recovery Time (ms)",
    ]
    df_params = pd.DataFrame({
        "Clinical Parameter": PARAM_LABELS,
        "Red Stimulus":  list(r_met),
        "Blue Stimulus": list(b_met),
    })

    with pd.ExcelWriter(export_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Raw Data", index=False)
        df_params.to_excel(writer, sheet_name="Clinical Parameters", index=False)


def export_metadata_json(state, export_path, eye_label, r_met, b_met):
    PARAM_LABELS = [
        "d_base", "d_min", "constrict_amp", "constrict_lat_ms",
        "max_cv", "avg_cv", "pipr", "max_rv", "recovery_75_ms",
    ]
    meta = {
        "patient_id":     state.patient_id,
        "diagnosis":      state.diagnosis,
        "eye":            eye_label,
        "timestamp_iso":  datetime.now().isoformat(),
        "left_enabled":   state.left_enabled,
        "right_enabled":  state.right_enabled,
        "save_mp4":       state.save_mp4,
        "clinical_metrics": {
            "red_stimulus":  dict(zip(PARAM_LABELS, r_met)),
            "blue_stimulus": dict(zip(PARAM_LABELS, b_met)),
        },
    }
    with open(export_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def export_all(state):
    """Full save: videos, PNG, HTML, XLSX, CSV, JSON metadata."""
    
    # ── MASTER OVERRIDE INTERCEPT ──
    # If Tracker Mode is active, bypass clinical processing and finalize immediately.
    if TrackerModeOverride.is_active(state):
        res = TrackerModeOverride.finalize_dataset(state)
        if res.get("ok"):
            # FIX: Manually clean up the session state since we are bypassing the standard cleanup block below
            with state.lock:
                state.has_staged_data = False
                state.staged_left = False
                state.staged_right = False
                # Unlink final paths so they cannot be accidentally deleted later
                state.stage_path_L = os.path.join(state.root_dir, ".temp_stage_L.avi")
                state.stage_path_R = os.path.join(state.root_dir, ".temp_stage_R.avi")
        return res
        
    if not state.has_staged_data or state.is_recording:
        return {"ok": False, "error": "Nothing staged or still recording."}
    if state.diagnosis in ("Select Diagnosis...", "", None):
        return {"ok": False, "error": "Select a diagnosis label before saving."}
    if len(state.patient_id) != 6 or not state.patient_id.isdigit():
        return {"ok": False, "error": "Invalid patient ID."}

    extracted_label = state.diagnosis.split(" ")[0]          
    date_str        = datetime.now().strftime("%m%d%Y")       
    pid             = state.patient_id.strip()
    
    trial_folder = f"{pid}_{date_str}"
    label_dir    = os.path.join(DATASET_ROOT, extracted_label, trial_folder)
    left_dir     = os.path.join(label_dir, "left")
    right_dir    = os.path.join(label_dir, "right")

    export_status = []

    elapsed_s  = state.elapsed()
    # UPDATED: Corrected the video length calculation for the 3-minute test.
    # We record exactly 60 seconds (Red phase) + the recorded portion of the Blue phase (120s to 180s).
    recorded_s = min(60, elapsed_s) + max(0, min(60, elapsed_s - 120))
    recorded_s = max(1.0, recorded_s)

    fps_L = max(1.0, state.frames_captured_L / recorded_s) if state.frames_captured_L > 0 else 30.0
    fps_R = max(1.0, state.frames_captured_R / recorded_s) if state.frames_captured_R > 0 else 30.0

    def _export_eye(eye, d_dir, prefix, stage_path, fps, session_data):
        os.makedirs(d_dir, exist_ok=True)

        _finalize_videos(stage_path,
                         os.path.join(d_dir, f"{prefix}.avi"),
                         os.path.join(d_dir, f"{prefix}.mp4"),
                         fps, state.save_mp4)
        export_status.append(f"{eye} Video")

        try:
            r_met, b_met = metrics.get_clinical_report(session_data)
        except Exception as e:
            err_msg = f"{eye} Metrics Engine FAILED: {e}"
            print(f"\n[EXPORTER ERROR] {err_msg}")
            export_status.append(err_msg)
            r_met, b_met = [0]*9, [0]*9

        try:
            export_graph_png(session_data,
                             os.path.join(d_dir, f"{prefix}.png"),
                             f"Patient {pid} ({eye} Eye)")
            export_status.append(f"{eye} PNG Plot")
        except Exception as e:
            err_msg = f"{eye} PNG FAILED: {e}"
            print(f"\n[EXPORTER ERROR] {err_msg}")
            export_status.append(err_msg)

        try:
            export_interactive_html(session_data,
                                    os.path.join(d_dir, f"{prefix}.html"),
                                    f"Patient {pid} ({eye} Eye)")
            export_status.append(f"{eye} HTML Plot")
        except Exception as e:
            err_msg = f"{eye} HTML FAILED: {e}"
            print(f"\n[EXPORTER ERROR] {err_msg}")
            export_status.append(err_msg)

        try:
            export_excel_report(session_data,
                                os.path.join(d_dir, f"{prefix}.xlsx"),
                                r_met, b_met)
            export_status.append(f"{eye} Excel")
        except Exception as e:
            err_msg = f"{eye} Excel FAILED: {e}"
            print(f"\n[EXPORTER ERROR] {err_msg}")
            export_status.append(err_msg)

        try:
            export_metadata_json(state,
                                 os.path.join(d_dir, f"{prefix}.json"),
                                 eye, r_met, b_met)
            export_status.append(f"{eye} Metadata JSON")
        except Exception as e:
            err_msg = f"{eye} JSON FAILED: {e}"
            print(f"\n[EXPORTER ERROR] {err_msg}")
            export_status.append(err_msg)

    if state.staged_left:
        _export_eye("Left",  left_dir,  f"{pid}-{date_str}-LE",
                    state.stage_path_L, fps_L, state.session_data["L"])

    if state.staged_right:
        _export_eye("Right", right_dir, f"{pid}-{date_str}-RE",
                    state.stage_path_R, fps_R, state.session_data["R"])

    try:
        state.pupil.save(
            patient_id=pid,
            date_str=date_str,
            left_dir=left_dir,
            right_dir=right_dir,
            staged_left=state.staged_left,
            staged_right=state.staged_right,
            diagnosis_label=extracted_label,
        )
        export_status.append("CSV Telemetry")
    except Exception as e:
        err_msg = f"CSV FAILED: {e}"
        print(f"\n[EXPORTER ERROR] {err_msg}")
        export_status.append(err_msg)

    with state.lock:
        state.has_staged_data = False
        state.staged_left     = False
        state.staged_right    = False
        state.patient_id      = ""
        state.diagnosis       = "Select Diagnosis..."

    return {"ok": True, "error": None, "status": export_status, "trial_folder": trial_folder}