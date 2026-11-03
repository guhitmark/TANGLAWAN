# pupil_tracker.py
# ---------------------------------------------------------------
# Hardware: HAMPO 003-0766 / OV9281 CMOS, 640x400 @ 30fps, IR.
# Backend : TensorRT FP16 engine (built by build_trt_engine.py)
# Switch  : tracker_config.TRACKER_ENABLED = True / False
# Mode    : CPU black-hat blob search -> native-resolution network
#           window (ROI_TRACKING), then 1 network pass per frame
#           while locked on
# Timing  : prints a [TIMING] line every 60 frames per tracker.
#           Set TIMING_PRINT = False in tracker_config.py to silence it.
# ---------------------------------------------------------------

import cv2
import numpy as np
import math
import os
import time
import threading
import ctypes
import glob
from dataclasses import dataclass
from typing import Optional

from controllers import tracker_config as cfg

cv2.setNumThreads(2)   # leave CPU cores for the camera/UI threads


# --- CUDA runtime via ctypes (no torch, no pycuda) -----------------
class _CudaRT:
    H2D, D2H = 1, 2   # cudaMemcpyHostToDevice / DeviceToHost

    def __init__(self):
        candidates = ["libcudart.so"] + sorted(glob.glob("/usr/local/cuda*/lib64/libcudart.so*"))
        self.lib = None
        for c in candidates:
            try:
                self.lib = ctypes.CDLL(c)
                break
            except OSError:
                continue
        if self.lib is None:
            raise RuntimeError("libcudart not found")
        L = self.lib
        vp, sz = ctypes.c_void_p, ctypes.c_size_t
        L.cudaMalloc.argtypes = [ctypes.POINTER(vp), sz]
        L.cudaHostAlloc.argtypes = [ctypes.POINTER(vp), sz, ctypes.c_uint]
        L.cudaMemcpyAsync.argtypes = [vp, vp, sz, ctypes.c_int, vp]
        L.cudaStreamCreate.argtypes = [ctypes.POINTER(vp)]
        L.cudaStreamSynchronize.argtypes = [vp]
        L.cudaSetDeviceFlags.argtypes = [ctypes.c_uint]
        L.cudaSetDeviceFlags(4)   # BlockingSync: sleep while waiting instead of spinning
        self.check(L.cudaFree(None), "init")      # forces context creation

    def check(self, err, what):
        if err != 0:
            raise RuntimeError("CUDA error %d during %s" % (err, what))

    def malloc(self, nbytes):
        p = ctypes.c_void_p()
        self.check(self.lib.cudaMalloc(ctypes.byref(p), nbytes), "cudaMalloc")
        return p.value

    def host_array(self, shape, dtype):
        """Pinned (page-locked) numpy array."""
        n = int(np.prod(shape)) * np.dtype(dtype).itemsize
        p = ctypes.c_void_p()
        self.check(self.lib.cudaHostAlloc(ctypes.byref(p), n, 0), "cudaHostAlloc")
        buf = (ctypes.c_char * n).from_address(p.value)
        return np.frombuffer(buf, dtype=dtype).reshape(shape)

    def stream_create(self):
        s = ctypes.c_void_p()
        self.check(self.lib.cudaStreamCreate(ctypes.byref(s)), "cudaStreamCreate")
        return s

    def copy(self, dst, src, nbytes, kind, stream):
        self.check(self.lib.cudaMemcpyAsync(dst, src, nbytes, kind, stream), "cudaMemcpyAsync")

    def sync(self, stream):
        self.check(self.lib.cudaStreamSynchronize(stream), "cudaStreamSynchronize")


class TRTSegmenter:
    def __init__(self, engine_path: str, w: int, h: int):
        import tensorrt as trt
        self.w, self.h = w, h
        self.lock = threading.Lock()
        self.rt = _CudaRT()

        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.ctx = self.engine.create_execution_context()

        self.h_in = self.rt.host_array((h, w), np.float32)
        self.h_out = self.rt.host_array((h, w), np.float32)
        self.d_in = self.rt.malloc(self.h_in.nbytes)
        self.d_out = self.rt.malloc(self.h_out.nbytes)
        self.stream = self.rt.stream_create()
        self._mask = np.empty((h, w), dtype=np.bool_)

        self.bindings = [0] * self.engine.num_bindings
        self.bindings[self.engine.get_binding_index("input")] = self.d_in
        self.bindings[self.engine.get_binding_index("logits")] = self.d_out

        self.t_gpu = 0.0      # seconds spent inside the lock on convert + copy + engine + sync
        self.n_gpu = 0
        self._infer(np.zeros((h, w), dtype=np.uint8))   # warm-up
        self.t_gpu, self.n_gpu = 0.0, 0                 # drop the warm-up from the stats

    def _infer(self, gray_u8: np.ndarray) -> np.ndarray:
        rt = self.rt
        with self.lock:
            t0 = time.perf_counter()
            np.multiply(gray_u8, 1.0 / 255.0, out=self.h_in, dtype=np.float32)
            rt.copy(self.d_in, self.h_in.ctypes.data, self.h_in.nbytes, rt.H2D, self.stream)
            self.ctx.execute_async_v2(self.bindings, self.stream.value)
            rt.copy(self.h_out.ctypes.data, self.d_out, self.h_out.nbytes, rt.D2H, self.stream)
            rt.sync(self.stream)
            self.t_gpu += time.perf_counter() - t0
            self.n_gpu += 1
            np.greater(self.h_out, 0, out=self._mask)     # logit > 0 == sigmoid > 0.5
            return self._mask.view(np.uint8).copy()

    def predict(self, gray_u8: np.ndarray) -> np.ndarray:
        return self._infer(gray_u8)


# --- GLOBAL SINGLETON ----------------------------------------------
_SHARED_MODEL = None
_MODEL_LOCK = threading.Lock()


def get_shared_model():
    """Returns the TRTSegmenter, or None if disabled / engine missing."""
    global _SHARED_MODEL
    if not getattr(cfg, "TRACKER_ENABLED", True):
        return None
    if _SHARED_MODEL is None:
        with _MODEL_LOCK:
            if _SHARED_MODEL is None:
                engine_path = os.path.join(os.path.dirname(__file__), cfg.ENGINE_FILE)
                if not os.path.exists(engine_path):
                    print(f"[AI] ERROR: {engine_path} not found. "
                          f"Build it with build_trt_engine.py on the Jetson and place it in controllers/.")
                    return None
                print("[AI] Loading TensorRT FP16 engine...")
                _SHARED_MODEL = TRTSegmenter(engine_path, cfg.INFER_W, cfg.INFER_H)
                print("[AI] Engine ready.")
    return _SHARED_MODEL


# --- Preprocessing helpers -----------------------------------------
def _letterbox(gray, W, H):
    """Uniform scale + edge-replicate padding. Returns (img, scale, pad_x, pad_y)."""
    fh, fw = gray.shape[:2]
    s = min(W / float(fw), H / float(fh))
    nw, nh = int(round(fw * s)), int(round(fh * s))
    interp = cv2.INTER_AREA if s < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(gray, (nw, nh), interpolation=interp)
    px, py = (W - nw) // 2, (H - nh) // 2
    out = cv2.copyMakeBorder(resized, py, H - nh - py, px, W - nw - px,
                             cv2.BORDER_REPLICATE)
    return out, s, px, py


def _dark_candidates(gray, n=3):
    """Find up to n dark blobs (darker than their surroundings) as (x, y) frame px.
    Black top-hat keeps dark features smaller than the kernel (pupil) and
    suppresses large dark areas (vignette, lens rim). The network decides
    which candidate is really a pupil."""
    k = 4                                              # work at 1/4 scale
    fh, fw = gray.shape[:2]
    small = cv2.resize(gray, (max(1, fw // k), max(1, fh // k)),
                       interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), 2.0)       # kills lashes / glints

    # Kernel must be larger than the biggest pupil (frame px)
    max_pupil = float(getattr(cfg, "SEARCH_PUPIL_MAX_PX", 130))
    ksz = int(round(max_pupil / k)) | 1                # odd
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksz, ksz))
    resp = cv2.morphologyEx(small, cv2.MORPH_BLACKHAT, kernel)

    sh, sw = resp.shape[:2]
    m = max(2, int(0.05 * min(sh, sw)))                # ignore the very edge
    resp[:m, :] = 0
    resp[-m:, :] = 0
    resp[:, :m] = 0
    resp[:, -m:] = 0

    sup_r = max(4, int(0.4 * cfg.INFER_W / k))         # suppression radius around each pick
    out = []
    for _ in range(n):
        _, max_val, _, max_loc = cv2.minMaxLoc(resp)
        if max_val < 12:                               # no meaningful dark blob left
            break
        out.append((max_loc[0] * k + k / 2.0, max_loc[1] * k + k / 2.0))
        cv2.circle(resp, max_loc, sup_r, 0, -1)
    return out


def _fit_circle(cnt, trim=0.8, rounds=3):
    """Round pupil from a contour: least-squares circle, then refit on the `trim` fraction of
    points that agree best. A half-hidden pupil (eyelid chord, glint notch, reflection) cannot
    stretch it into a long ellipse. Returns (cx, cy, r) or None."""
    p = cnt.reshape(-1, 2).astype(np.float64)
    if len(p) < 5:
        return None
    keep = p
    cx = cy = r = float("nan")
    for _ in range(rounds):
        A = np.column_stack((2.0 * keep[:, 0], 2.0 * keep[:, 1], np.ones(len(keep))))
        b = (keep ** 2).sum(axis=1)
        try:
            sol = np.linalg.lstsq(A, b, rcond=-1)[0]    # rcond=-1: works on old numpy (Jetson) too
        except Exception:
            return None
        cx, cy, c = float(sol[0]), float(sol[1]), float(sol[2])
        r2 = c + cx * cx + cy * cy
        if not (r2 > 0.0):
            return None
        r = math.sqrt(r2)
        err = np.abs(np.hypot(p[:, 0] - cx, p[:, 1] - cy) - r)
        n = max(5, int(trim * len(p)))
        keep = p[np.argsort(err)[:n]]
    if not (math.isfinite(cx) and math.isfinite(cy) and r > 1.0):
        return None
    return float(cx), float(cy), float(r)


@dataclass
class PupilResult:
    cx_px: float = float("nan")
    cy_px: float = float("nan")
    major_px: float = float("nan")
    minor_px: float = float("nan")
    diameter_px: float = float("nan")
    diameter_mm: float = float("nan")
    angle_deg: float = float("nan")
    confidence: float = 0.0
    ir_mode: str = "ai_unet"
    frame_index: int = 0
    timestamp_s: float = 0.0

    @property
    def is_valid(self) -> bool:
        return not math.isnan(self.diameter_px) and self.confidence >= 0.3


class PupilTracker:
    def __init__(self, mm_per_pixel: Optional[float] = None):
        self.mm_per_pixel = mm_per_pixel if mm_per_pixel else cfg.MM_PER_PIXEL_FALLBACK
        self.enabled = bool(getattr(cfg, "TRACKER_ENABLED", True))
        self.model = get_shared_model() if self.enabled else None
        self._center = None      # last pupil center (frame px), for ROI tracking
        # timing counters (reset every 60 frames)
        self._t_total = 0.0      # whole process() call
        self._t_net = 0.0        # time inside model.predict() incl. waiting for the shared lock
        self._n_frames = 0
        self._n_passes = 0
        
        self.alpha = float(getattr(cfg, "SMOOTHING_ALPHA", 0.15))
        self._smooth_geom = None 
        self._smooth_diam = None
        
        if not self.enabled:
            print("[AI] TRACKER_ENABLED = False -> tracker not loaded.")

    def _segment(self, gray, center=None):
        """One network pass.
        center=None   : whole frame (letterboxed to the engine size).
        center=(x, y) : native-resolution window around that point.
        Returns (cx, cy, MA, ma, angle) in original-frame pixels, or None."""
        fh, fw = gray.shape[:2]
        W, H = cfg.INFER_W, cfg.INFER_H

        if center is not None:
            x0 = int(round(center[0] - W / 2.0))
            y0 = int(round(center[1] - H / 2.0))
            x0 = max(0, min(fw - W, x0))
            y0 = max(0, min(fh - H, y0))
            img = np.ascontiguousarray(gray[y0:y0 + H, x0:x0 + W])
            s, px, py, ox, oy = 1.0, 0, 0, x0, y0
        else:
            if fw == W and fh == H:
                img = np.ascontiguousarray(gray)
                s, px, py = 1.0, 0, 0
            else:
                img, s, px, py = _letterbox(gray, W, H)
            ox = oy = 0

        t_net = time.perf_counter()
        mask = self.model.predict(img)                 # uint8 0/1
        self._t_net += time.perf_counter() - t_net
        self._n_passes += 1
        cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]

        min_area = 100.0 * s * s                       # "100 px" in original-frame terms
        best, largest = None, 0.0
        for cnt in cnts:
            if len(cnt) >= 5:
                area = cv2.contourArea(cnt)
                if area > min_area and area > largest:
                    largest, best = area, cnt
        if best is None:
            return None

        if center is not None:
            # Pupil touching the window edge = window is off target
            x, y, w, h = cv2.boundingRect(best)
            if x <= 0 or y <= 0 or x + w >= W or y + h >= H:
                return None

        (cx, cy), (MA, ma), angle = cv2.fitEllipse(best)
        mode = getattr(cfg, "PUPIL_FIT", "ellipse")

        if mode == "circle":
            # Legacy hard-circle mode: collapse to equal-area circle
            circ = _fit_circle(best)
            if circ is not None:
                cx, cy, r = circ
                MA = ma = 2.0 * r
            else:
                MA = ma = 2.0 * math.sqrt(largest / math.pi)
            angle = 0.0

        else:
            # "ellipse" mode (default) or any unrecognised value:
            # Accept the fitted ellipse but cap the axis ratio so the result
            # stays physiologically plausible.  Human pupils viewed head-on
            # rarely exceed ~1.3 : 1; at mild off-axis gaze up to ~1.5 : 1.
            # Beyond that the fit is almost certainly chasing a noise artefact
            # (lash, glint, partial occlusion) rather than the real pupil edge.
            MAX_RATIO = float(getattr(cfg, "PUPIL_MAX_RATIO", 1.5))
            # ensure MA is always the long axis
            if MA < ma:
                MA, ma = ma, MA
                angle = (angle + 90.0) % 180.0
            ratio = MA / max(1e-6, ma)
            if ratio > MAX_RATIO:
                # Clamp: preserve the long axis, grow the short axis up to meet the cap.
                # This keeps the centre and orientation intact while reducing eccentricity.
                ma = MA / MAX_RATIO

        return ((cx - px) / s + ox, (cy - py) / s + oy, MA / s, ma / s, angle)

    def process(self, frame: np.ndarray, frame_index: int = 0,
                timestamp_s: float = 0.0) -> PupilResult:
        t_start = time.perf_counter()
        result = PupilResult(frame_index=frame_index, timestamp_s=timestamp_s)

        if frame is None or frame.size == 0 or self.model is None:
            return result

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        fh, fw = gray.shape[:2]
        can_roi = (getattr(cfg, "ROI_TRACKING", True)
                   and fw >= cfg.INFER_W and fh >= cfg.INFER_H)

        res = None
        if can_roi:
            # 1) locked on: one network pass around the last position
            if self._center is not None:
                res = self._segment(gray, self._center)
            # 2) lost / first frame: test the dark blobs, network decides
            if res is None:
                n = int(getattr(cfg, "COARSE_CANDIDATES", 3))
                cands = _dark_candidates(gray, n)
                cands.append((fw / 2.0, fh / 2.0))      # fallback: frame center
                for c in cands:
                    res = self._segment(gray, c)
                    if res is not None:
                        break
        else:
            # Engine size == full frame (or ROI disabled): classic whole-frame pass
            res = self._segment(gray)

        self._center = (res[0], res[1]) if res is not None else None

        if res is not None:
            cx, cy, MA, ma, angle = res
            
            # --- APPLY EXPONENTIAL MOVING AVERAGE (EMA) FILTER ---
            if self._smooth_geom is None:
                self._smooth_geom = [cx, cy, MA, ma]
            else:
                self._smooth_geom[0] += self.alpha * (cx - self._smooth_geom[0])
                self._smooth_geom[1] += self.alpha * (cy - self._smooth_geom[1])
                self._smooth_geom[2] += self.alpha * (MA - self._smooth_geom[2])
                self._smooth_geom[3] += self.alpha * (ma - self._smooth_geom[3])
                
            s_cx, s_cy, s_MA, s_ma = self._smooth_geom
            self._smooth_diam = (s_MA + s_ma) / 2.0
            # -----------------------------------------------------

            result.cx_px = round(s_cx, 2)
            result.cy_px = round(s_cy, 2)
            result.major_px = round(s_MA / 2.0, 2)
            result.minor_px = round(s_ma / 2.0, 2)
            result.diameter_px = round(self._smooth_diam, 2)
            result.angle_deg = round(angle, 2)
            result.confidence = 0.95
            result.diameter_mm = round(result.diameter_px * self.mm_per_pixel,
                                       cfg.CSV_FLOAT_PRECISION)
        else:
            # Reset the filter if the tracker loses the eye
            self._smooth_geom = None
            self._smooth_diam = None

        self._t_total += time.perf_counter() - t_start
        self._n_frames += 1
        if self._n_frames >= 60:
            if getattr(cfg, "TIMING_PRINT", True):
                total_ms = 1000.0 * self._t_total / self._n_frames
                net_ms = 1000.0 * self._t_net / max(1, self._n_passes)
                gpu_ms = 1000.0 * self.model.t_gpu / max(1, self.model.n_gpu)
                print("[TIMING] frame %.1f ms | net call %.1f ms x %.2f/frame | "
                      "engine only %.1f ms | everything else %.1f ms"
                      % (total_ms, net_ms, self._n_passes / float(self._n_frames), gpu_ms,
                         total_ms - net_ms * self._n_passes / float(self._n_frames)))
            self._t_total = self._t_net = 0.0
            self._n_frames = self._n_passes = 0
            self.model.t_gpu, self.model.n_gpu = 0.0, 0
        return result

    def update_calibration(self, mm_per_pixel: float):
        self.mm_per_pixel = mm_per_pixel

    def draw_result(self, frame: np.ndarray, result: PupilResult) -> np.ndarray:
        out = frame.copy() if frame.ndim == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        GREEN = (0, 220, 80)

        if not self.enabled:
            cv2.putText(out, "[ TRACKER OFF ]", (15, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 165, 255), 2, cv2.LINE_AA)
            return out

        if not result.is_valid:
            cv2.putText(out, "[ TRACKING LOST ]", (15, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 40, 200), 2, cv2.LINE_AA)
            return out

        cx, cy = int(result.cx_px), int(result.cy_px)
        r_maj = max(int(result.major_px), 4)
        r_min = max(int(result.minor_px), 4)
        cv2.ellipse(out, (cx, cy), (r_maj, r_min), result.angle_deg, 0, 360, GREEN, 1, cv2.LINE_AA)
        cv2.drawMarker(out, (cx, cy), GREEN, cv2.MARKER_CROSS, 8, 1)
        readout = f"D {result.diameter_px:.1f}px  {result.diameter_mm:.2f}mm  CF {result.confidence:.2f}"
        cv2.putText(out, readout, (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, GREEN, 2, cv2.LINE_AA)
        return out