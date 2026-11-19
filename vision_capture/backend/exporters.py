"""exporters.py"""

import os
import json
from datetime import datetime

import numpy as np
import cv2

from . import metrics
from .mode_manager import TrackerModeOverride

# ── Fixed dataset root ────────────────────────────────────────────
DATASET_ROOT = "/home/tanglawan/Desktop/vision_capture_v1/dataset"
from . import plr_stabilizer

def _finalize_videos(temp_avi, final_avi, actual_fps):
    if not os.path.exists(temp_avi):
        return
    cap = cv2.VideoCapture(temp_avi)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_avi = cv2.VideoWriter(final_avi, cv2.VideoWriter_fourcc(*"MJPG"), actual_fps, (w, h))

    try:
        while True:
            ret, frame = cap.read()
            if not ret: break
            out_avi.write(frame)
    finally:
        cap.release()
        out_avi.release()
        
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

    series = plr_stabilizer.plot_series(plr_stabilizer.stabilize_session(session_data_eye))
    for ax, key, color in [(ax_red, "red", "#dc2626"), (ax_blue, "blue", "#2563eb")]:
        q = series[key]
        ax.plot(q["t"], q["measured"], color=color, lw=1.5, label="Filtered measurement")
        ax.plot(q["t"], q["held"], color=color, lw=1.3, ls="--", label="Last valid value (held)")
        if len(q["held_t"]):
            ax.scatter(q["held_t"], q["held_y"], color="#f59e0b", marker="x", s=10,
                       alpha=0.65, label="Lost / artifact", zorder=5)
        if len(q["missing_t"]):
            ax.plot(q["missing_t"], np.full(len(q["missing_t"]), 0.025), "|", color="#f59e0b",
                    transform=ax.get_xaxis_transform(), label="No previous measurement")

    for ax in (ax_red, ax_blue):
        ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(export_path, bbox_inches="tight")
    plt.close(fig)


def export_interactive_html(session_data_eye, export_path, title):
    import plotly.graph_objects as go

    if not session_data_eye:
        raise ValueError("No telemetry data to plot.")

    series = plr_stabilizer.plot_series(plr_stabilizer.stabilize_session(session_data_eye))
    fig = go.Figure()
    for key, name, color in [("red", "Red", "#dc2626"), ("blue", "Blue", "#2563eb")]:
        q = series[key]
        fig.add_trace(go.Scatter(x=q["t"], y=q["measured"], mode="lines", connectgaps=False,
                     name=name + " filtered", line=dict(color=color, width=2)))
        fig.add_trace(go.Scatter(x=q["t"], y=q["held"], mode="lines", connectgaps=False,
                     name=name + " held (not measured)", line=dict(color=color, width=2, dash="dash")))
        if len(q["held_t"]):
            fig.add_trace(go.Scatter(x=q["held_t"], y=q["held_y"], mode="markers",
                         name=name + " lost / artifact", marker=dict(color="#f59e0b", symbol="x", size=5)))

    fig.update_layout(
        title=f"Interactive PLR — {title}",
        xaxis_title="Time (s)", yaxis_title="Pupil Diameter (mm)",
        template="plotly_dark", hovermode="x unified", dragmode="zoom",
        yaxis=dict(range=[1.0, 9.0]),
        xaxis=dict(range=[0, 180], rangeslider=dict(visible=True, thickness=0.08)),
    )
    fig.write_html(export_path, config={"scrollZoom": True, "displayModeBar": True})



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
        "save_mp4":       False,
        "clinical_metrics_source": "Existing metrics engine; not recomputed from display holds",
        "clinical_metrics": {
            "red_stimulus":  dict(zip(PARAM_LABELS, r_met)),
            "blue_stimulus": dict(zip(PARAM_LABELS, b_met)),
        },
    }
    with open(export_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def export_all(state):
    """Full save: AVI, PNG, HTML, CSV and JSON metadata."""
    # Keep the backend flag disabled even if an older UI enabled MP4.
    state.save_mp4 = False
    
    if TrackerModeOverride.is_active(state):
        res = TrackerModeOverride.finalize_dataset(state)
        if res.get("ok"):
            with state.lock:
                state.has_staged_data = False
                state.staged_left = False
                state.staged_right = False
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
    recorded_s = min(60, elapsed_s) + max(0, min(60, elapsed_s - 120))
    recorded_s = max(1.0, recorded_s)

    fps_L = max(1.0, state.frames_captured_L / recorded_s) if state.frames_captured_L > 0 else 30.0
    fps_R = max(1.0, state.frames_captured_R / recorded_s) if state.frames_captured_R > 0 else 30.0

    def _export_eye(eye, d_dir, prefix, stage_path, fps, session_data):
        os.makedirs(d_dir, exist_ok=True)

        _finalize_videos(stage_path,
                         os.path.join(d_dir, f"{prefix}.avi"),
                         fps)
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
        # Add derived fields only; never overwrite the original diameter or valid columns.
        for enabled, eye_dir, suffix in [(state.staged_left, left_dir, "LE"),
                                         (state.staged_right, right_dir, "RE")]:
            if not enabled:
                continue
            candidates = [os.path.join(eye_dir, f"{date_str}-{pid}-{suffix}_pupil.csv"),
                          os.path.join(eye_dir, f"{pid}-{date_str}-{suffix}_pupil.csv")]
            found = [path for path in candidates if os.path.isfile(path)]
            if len(found) != 1:
                export_status.append(f"{suffix} stabilizer skipped: expected exactly one pupil CSV")
                continue
            try:
                plr_stabilizer.add_filtered_csv(found[0])
                export_status.append(f"{suffix} filtered CSV columns")
            except Exception as e:
                export_status.append(f"{suffix} stabilizer FAILED: {e}")
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