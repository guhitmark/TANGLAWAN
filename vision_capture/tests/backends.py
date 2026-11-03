"""backends.py - shared helpers for accuracy / efficiency testing of the pupil tracker.

Runs on the Jetson (Python 3.6 compatible). torch / tensorrt are imported lazily,
so the numpy-only helpers can also be imported on a PC.
Place next to run.py (project root) so `from controllers import ...` works.
"""
import os

import cv2
import numpy as np

NAN = float("nan")
DEFAULT_WIDTHS = (16, 32, 64, 128, 256)


# ------------------------------------------------------------------
# System memory (Linux / Jetson). Returns NaN where unavailable.
# ------------------------------------------------------------------
def rss_mb():
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS"):
                    return int(line.split()[1]) / 1024.0
    except (IOError, OSError):
        pass
    return NAN


def system_used_mb():
    """Total - available. On Jetson the GPU shares this RAM, so it includes GPU memory."""
    try:
        d = {}
        with open("/proc/meminfo") as f:
            for line in f:
                k, v = line.split(":")
                d[k] = int(v.split()[0])
        return (d["MemTotal"] - d["MemAvailable"]) / 1024.0
    except (IOError, OSError, KeyError):
        return NAN


# ------------------------------------------------------------------
# Statistics
# ------------------------------------------------------------------
def summarize(values):
    a = np.asarray(values, dtype=np.float64)
    a = a[~np.isnan(a)]
    if a.size == 0:
        return {"n": 0}
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "sd": float(a.std(ddof=1)) if a.size > 1 else 0.0,
        "median": float(np.median(a)),
        "p5": float(np.percentile(a, 5)),
        "p95": float(np.percentile(a, 95)),
        "p99": float(np.percentile(a, 99)),
        "min": float(a.min()),
        "max": float(a.max()),
    }


# ------------------------------------------------------------------
# Segmentation / geometry (same rules as the app's tracker)
# ------------------------------------------------------------------
def seg_metrics(pred_u8, gt_u8):
    p = pred_u8.astype(bool)
    g = gt_u8.astype(bool)
    tp = float(np.logical_and(p, g).sum())
    fp = float(np.logical_and(p, ~g).sum())
    fn = float(np.logical_and(~p, g).sum())
    tn = float(p.size) - tp - fp - fn
    d_dice = 2 * tp + fp + fn
    d_iou = tp + fp + fn
    return {
        "dice": 2 * tp / d_dice if d_dice > 0 else NAN,
        "iou": tp / d_iou if d_iou > 0 else NAN,
        "precision": tp / (tp + fp) if (tp + fp) > 0 else NAN,
        "recall": tp / (tp + fn) if (tp + fn) > 0 else NAN,
        "pixel_acc": (tp + tn) / float(p.size),
        "pred_area": float(p.sum()),
        "gt_area": float(g.sum()),
    }


def mask_to_ellipse(mask_u8, min_area=100.0):
    """Largest contour -> (cx, cy, MA, ma, angle, diameter) or None.
    Same rule as PupilTracker: area > min_area, >= 5 points, fitEllipse,
    diameter = (MA + ma) / 2."""
    cnts = cv2.findContours(mask_u8.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    best, largest = None, 0.0
    for c in cnts:
        if len(c) >= 5:
            a = cv2.contourArea(c)
            if a > min_area and a > largest:
                largest, best = a, c
    if best is None:
        return None
    (cx, cy), (MA, ma), ang = cv2.fitEllipse(best)
    return cx, cy, MA, ma, ang, (MA + ma) / 2.0


def list_pairs(test_dir):
    """(name, image_path, mask_path) for <test_dir>/images/X.jpg + <test_dir>/masks/X_mask.png"""
    img_dir = os.path.join(test_dir, "images")
    mask_dir = os.path.join(test_dir, "masks")
    pairs = []
    for name in sorted(os.listdir(img_dir)):
        stem, ext = os.path.splitext(name)
        if ext.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
            continue
        mp = os.path.join(mask_dir, stem + "_mask.png")
        if os.path.exists(mp):
            pairs.append((name, os.path.join(img_dir, name), mp))
    return pairs


# ------------------------------------------------------------------
# Analytic complexity of the Slim U-Net (batch 1)
# ------------------------------------------------------------------
def analytic_macs(h, w, c=DEFAULT_WIDTHS, cin=1, cout=1):
    def conv(ci, co, hh, ww):
        return ci * co * 9 * hh * ww

    def dconv(ci, co, hh, ww):
        return conv(ci, co, hh, ww) + conv(co, co, hh, ww)

    def deconv(ci, co, hin, win):
        return ci * co * 4 * hin * win

    t = dconv(cin, c[0], h, w)
    t += dconv(c[0], c[1], h // 2, w // 2)
    t += dconv(c[1], c[2], h // 4, w // 4)
    t += dconv(c[2], c[3], h // 8, w // 8)
    t += dconv(c[3], c[4], h // 16, w // 16)
    t += deconv(c[4], c[3], h // 16, w // 16) + dconv(c[4], c[3], h // 8, w // 8)
    t += deconv(c[3], c[2], h // 8, w // 8) + dconv(c[3], c[2], h // 4, w // 4)
    t += deconv(c[2], c[1], h // 4, w // 4) + dconv(c[2], c[1], h // 2, w // 2)
    t += deconv(c[1], c[0], h // 2, w // 2) + dconv(c[1], c[0], h, w)
    t += c[0] * cout * h * w
    return t


def analytic_params(c=DEFAULT_WIDTHS, cin=1, cout=1):
    """Trainable parameters (conv weights, BN scale/shift, deconv weights+bias, final conv)."""
    def dconv(ci, co):
        return ci * co * 9 + 2 * co + co * co * 9 + 2 * co

    def deconv(ci, co):
        return ci * co * 4 + co

    t = dconv(cin, c[0]) + dconv(c[0], c[1]) + dconv(c[1], c[2]) + dconv(c[2], c[3]) + dconv(c[3], c[4])
    t += deconv(c[4], c[3]) + dconv(c[4], c[3])
    t += deconv(c[3], c[2]) + dconv(c[3], c[2])
    t += deconv(c[2], c[1]) + dconv(c[2], c[1])
    t += deconv(c[1], c[0]) + dconv(c[1], c[0])
    t += c[0] * cout + cout
    return t


# ------------------------------------------------------------------
# Inference backends: predict(gray_u8 HxW) -> uint8 0/1 mask HxW
# ------------------------------------------------------------------
def build_unet(widths=None):
    import torch.nn as nn
    import torch

    def cbr(i, o, s=1):
        return nn.Sequential(nn.Conv2d(i, o, 3, stride=s, padding=1, bias=False),
                             nn.BatchNorm2d(o), nn.ReLU(inplace=True))

    class DoubleConv(nn.Module):
        def __init__(self, i, o):
            super().__init__()
            self.conv = nn.Sequential(cbr(i, o), cbr(o, o))

        def forward(self, x):
            return self.conv(x)

    class LitePupilNet(nn.Module):
        def __init__(self, c=(12, 24, 48, 96), d=4):
            super().__init__()
            self.detail = cbr(1, d)
            self.stem = cbr(1, c[0], s=2)
            self.e1 = cbr(c[0], c[0])
            self.e2 = DoubleConv(c[0], c[1])
            self.e3 = DoubleConv(c[1], c[2])
            self.e4 = DoubleConv(c[2], c[3])
            self.pool = nn.MaxPool2d(2, 2)
            self.u3 = nn.ConvTranspose2d(c[3], c[2], 2, 2); self.d3 = cbr(c[2], c[2])
            self.u2 = nn.ConvTranspose2d(c[2], c[1], 2, 2); self.d2 = cbr(c[1], c[1])
            self.u1 = nn.ConvTranspose2d(c[1], c[0], 2, 2); self.d1 = cbr(c[0], c[0])
            self.u0 = nn.ConvTranspose2d(c[0], 8, 2, 2)
            self.head = nn.Sequential(cbr(8 + d, 8), nn.Conv2d(8, 1, 1))

        def forward(self, x):
            det = self.detail(x)
            s1 = self.e1(self.stem(x))
            s2 = self.e2(self.pool(s1))
            s3 = self.e3(self.pool(s2))
            b = self.e4(self.pool(s3))
            y = self.d3(self.u3(b) + s3)
            y = self.d2(self.u2(y) + s2)
            y = self.d1(self.u1(y) + s1)
            return self.head(torch.cat((self.u0(y), det), 1))

    # Note: Using the "lite" variant widths (12, 24, 48, 96). 
    # If your best model was the "tiny" variant, change this to c=(8, 16, 32, 64)
    return LitePupilNet(c=(12, 24, 48, 96))


class TRTBackend(object):
    """The deployed path: TRTSegmenter from controllers/pupil_tracker.py."""
    def __init__(self, engine_path, w, h):
        from controllers.pupil_tracker import TRTSegmenter
        self.name = "trt_fp16"
        self.seg = TRTSegmenter(engine_path, w, h)

    def predict(self, gray_u8):
        return self.seg.predict(gray_u8)


class TorchBackend(object):
    """PyTorch eager on the GPU. half=False -> FP32 reference, half=True -> FP16."""
    def __init__(self, weights_path, w, h, half, widths):
        import torch
        self.torch = torch
        self.name = "torch_fp16" if half else "torch_fp32"
        self.half = half
        self.dev = torch.device("cuda")
        model = build_unet(widths)
        model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        model = model.to(self.dev).eval()
        if half:
            model = model.half()
        self.model = model
        self.predict(np.zeros((h, w), dtype=np.uint8))   # warm-up

    def predict(self, gray_u8):
        torch = self.torch
        with torch.no_grad():
            x = torch.from_numpy(np.ascontiguousarray(gray_u8)).to(self.dev)
            x = x.half() if self.half else x.float()
            x = (x / 255.0).unsqueeze(0).unsqueeze(0)
            out = self.model(x)
            return (out > 0)[0, 0].to(torch.uint8).cpu().numpy()


class DummyBackend(object):
    """Pipeline self-test only (no GPU, not a real model): dark-pixel threshold."""
    def __init__(self):
        self.name = "dummy"

    def predict(self, gray_u8):
        return (gray_u8 < 80).astype(np.uint8)


def make_backend(kind, engine=None, weights=None, w=400, h=400, widths=DEFAULT_WIDTHS):
    if kind == "trt":
        return TRTBackend(os.path.abspath(engine), w, h)
    if kind == "torch32":
        return TorchBackend(weights, w, h, False, widths)
    if kind == "torch16":
        return TorchBackend(weights, w, h, True, widths)
    if kind == "dummy":
        return DummyBackend()
    raise ValueError("unknown backend: %s" % kind)
