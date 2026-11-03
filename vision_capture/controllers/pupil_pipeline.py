# pupil_pipeline.py
# ---------------------------------------------------------------
# UNCHANGED from your uploaded version -- no Tkinter dependency
# existed here, so this just moves as-is into controllers/.
# ---------------------------------------------------------------

import csv
import math
import os
import time
import logging
from typing import Optional

import cv2
import numpy as np

from controllers import tracker_config as cfg
from controllers.pupil_tracker import PupilTracker, PupilResult


class EyePipeline:
    def __init__(self, eye_label: str, mm_per_pixel: Optional[float] = None):
        self.label    = eye_label   # "LE" or "RE"
        self.tracker  = PupilTracker(mm_per_pixel=mm_per_pixel)

        self._records: list = []
        self._frame_index: int = 0
        self._session_start: float = time.time()
        self._last_valid_diameter: float = float("nan")
        self._ema_diameter: float = float("nan")

    def reset(self):
        self._records = []
        self._frame_index = 0
        self._session_start = time.time()
        self._last_valid_diameter = float("nan")
        self._ema_diameter = float("nan")

    def process(self, frame: np.ndarray) -> PupilResult:
        ts = time.time() - self._session_start
        raw = self.tracker.process(frame, frame_index=self._frame_index, timestamp_s=ts)

        if raw.is_valid:
            self._last_valid_diameter = raw.diameter_px

            alpha = cfg.SMOOTHING_ALPHA
            if math.isnan(self._ema_diameter):
                self._ema_diameter = raw.diameter_px
            else:
                self._ema_diameter = alpha * raw.diameter_px + (1.0 - alpha) * self._ema_diameter

            smoothed_px = round(self._ema_diameter, cfg.CSV_FLOAT_PRECISION)
            smoothed_mm = round(smoothed_px * self.tracker.mm_per_pixel, cfg.CSV_FLOAT_PRECISION)
        else:
            fill = float("nan") if cfg.NAN_ON_LOSS else 0.0
            smoothed_px = fill
            smoothed_mm = fill

        self._records.append({
            "frame":          self._frame_index,
            "timestamp_s":    round(ts, 6),
            "diameter_px":    raw.diameter_px,
            "diameter_mm":    raw.diameter_mm,
            "smoothed_px":    smoothed_px,
            "smoothed_mm":    smoothed_mm,
            "cx_px":          raw.cx_px,
            "cy_px":          raw.cy_px,
            "major_px":       raw.major_px,
            "minor_px":       raw.minor_px,
            "angle_deg":      raw.angle_deg,
            "confidence":     raw.confidence,
            "ir_mode":        raw.ir_mode,
            "valid":          int(raw.is_valid),
        })

        self._frame_index += 1
        return raw

    def write_csv(self, csv_path: str, diagnosis_label: str = "UNLABELED"):
        """Flush buffered records to a .csv sidecar file with ground truth class."""
        if not self._records:
            return

        for r in self._records:
            r["ground_truth"] = diagnosis_label

        fieldnames = list(self._records[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=cfg.CSV_DELIMITER)
            writer.writeheader()
            writer.writerows(self._records)

    @property
    def diameter_array(self) -> list:
        return [r["smoothed_px"] for r in self._records]


class PupilPipeline:
    def __init__(
        self,
        mm_per_pixel_L: Optional[float] = None,
        mm_per_pixel_R: Optional[float] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self._pipe_L = EyePipeline("LE", mm_per_pixel=mm_per_pixel_L)
        self._pipe_R = EyePipeline("RE", mm_per_pixel=mm_per_pixel_R)
        self.logger  = logger or logging.getLogger("TanglawanPupil")

    def start_session(self):
        self._pipe_L.reset()
        self._pipe_R.reset()
        self.logger.info("PupilPipeline: session buffers reset.")

    def process_frame(self, frame: np.ndarray, eye: str) -> PupilResult:
        pipe = self._pipe_L if eye == "L" else self._pipe_R
        return pipe.process(frame)

    def draw(self, frame: np.ndarray, result: PupilResult, eye: str) -> np.ndarray:
        pipe = self._pipe_L if eye == "L" else self._pipe_R
        return pipe.tracker.draw_result(frame, result)

    def save(
        self,
        patient_id: str,
        date_str: str,
        left_dir: str,
        right_dir: str,
        staged_left: bool = True,
        staged_right: bool = True,
        diagnosis_label: str = "UNLABELED"
    ):
        if staged_left:
            path = os.path.join(left_dir, f"{date_str}-{patient_id}-LE_pupil.csv")
            self._pipe_L.write_csv(path, diagnosis_label)
            self.logger.info(f"PupilPipeline: LEFT  sidecar written -> {path}")

        if staged_right:
            path = os.path.join(right_dir, f"{date_str}-{patient_id}-RE_pupil.csv")
            self._pipe_R.write_csv(path, diagnosis_label)
            self.logger.info(f"PupilPipeline: RIGHT sidecar written -> {path}")

    def update_calibration(self, mm_per_pixel: float, eye: str = "both"):
        if eye in ("L", "both"):
            self._pipe_L.tracker.update_calibration(mm_per_pixel)
        if eye in ("R", "both"):
            self._pipe_R.tracker.update_calibration(mm_per_pixel)
        self.logger.info(f"PupilPipeline: calibration updated ({eye}) -> {mm_per_pixel:.5f} mm/px")

    @property
    def left_diameters(self) -> list:
        return self._pipe_L.diameter_array

    @property
    def right_diameters(self) -> list:
        return self._pipe_R.diameter_array