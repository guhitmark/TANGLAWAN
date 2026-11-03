"""bench_efficiency.py - latency, throughput, memory and complexity of one backend.

Run each backend in its OWN process (so memory numbers are not contaminated):

  python3 bench_efficiency.py --backend trt --engine controllers/pupil_tracker_fp16.engine \
          --images ~/data/annotated/test/images --n 1500 --two-eyes --out results/bench_trt.json

Measures (per frame, batch 1, after warm-up):
  predict_ms : preprocessing + host->device + inference + device->host + threshold
  post_ms    : contour + ellipse fit (CPU)
  total_ms   : predict + post  (what the tracker costs per frame)
--two-eyes also runs two threads sharing the backend (like the app: left + right eye).
"""
import argparse
import json
import os
import platform
import subprocess
import sys
import threading
import time

import cv2
import numpy as np

from backends import (make_backend, mask_to_ellipse, summarize, rss_mb, system_used_mb,
                      analytic_macs, analytic_params, DEFAULT_WIDTHS)


def load_images(img_dir, limit, w, h):
    names = [n for n in sorted(os.listdir(img_dir))
             if os.path.splitext(n)[1].lower() in (".jpg", ".jpeg", ".png", ".bmp")]
    imgs = []
    for n in names[:limit]:
        im = cv2.imread(os.path.join(img_dir, n), cv2.IMREAD_GRAYSCALE)
        if im is not None and im.shape == (h, w):
            imgs.append(np.ascontiguousarray(im))
    return imgs


def run_single(backend, imgs, n, warmup):
    pred_ms, post_ms = [], []
    k = len(imgs)
    for i in range(warmup + n):
        img = imgs[i % k]
        t0 = time.perf_counter()
        mask = backend.predict(img)
        t1 = time.perf_counter()
        mask_to_ellipse(mask)
        t2 = time.perf_counter()
        if i >= warmup:
            pred_ms.append((t1 - t0) * 1000.0)
            post_ms.append((t2 - t1) * 1000.0)
    total = [a + b for a, b in zip(pred_ms, post_ms)]
    return pred_ms, post_ms, total


def run_two_eyes(backend, imgs, n):
    k = len(imgs)
    barrier = threading.Barrier(2)
    walls = {}

    def worker(tag, offset):
        for i in range(20):
            backend.predict(imgs[(i + offset) % k])
        barrier.wait()
        t0 = time.perf_counter()
        for i in range(n):
            mask_to_ellipse(backend.predict(imgs[(i + offset) % k]))
        walls[tag] = time.perf_counter() - t0

    ts = [threading.Thread(target=worker, args=("L", 0)),
          threading.Thread(target=worker, args=("R", k // 2))]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    wall = max(walls.values())
    return {"frames_per_eye": n, "wall_s": wall,
            "fps_per_eye": n / wall, "fps_combined": 2.0 * n / wall}


def env_info():
    info = {"python": platform.python_version(), "machine": platform.machine()}
    for mod in ("torch", "tensorrt", "cv2", "numpy"):
        try:
            m = __import__(mod)
            info[mod] = getattr(m, "__version__", "?")
        except Exception:
            info[mod] = None
    try:
        out = subprocess.check_output(["nvpmodel", "-q"], stderr=subprocess.STDOUT).decode().strip()
        info["nvpmodel"] = out.replace("\n", " | ")
    except Exception:
        info["nvpmodel"] = None
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True, choices=["trt", "torch32", "torch16", "dummy"])
    ap.add_argument("--engine", default="controllers/pupil_tracker_fp16.engine")
    ap.add_argument("--weights", default="controllers/pupil_tracker_best.pth")
    ap.add_argument("--widths", type=int, nargs=5, default=list(DEFAULT_WIDTHS))
    ap.add_argument("--size", type=int, nargs=2, default=[400, 400], metavar=("W", "H"))
    ap.add_argument("--images", required=True, help="folder of test images (400x400 grayscale)")
    ap.add_argument("--n", type=int, default=1500, help="timed frames")
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--two-eyes", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    W, H = args.size
    imgs = load_images(args.images, 200, W, H)
    if not imgs:
        sys.exit("No %dx%d images found in %s" % (W, H, args.images))

    # ---- memory footprint of the backend (import + CUDA init + model + buffers)
    used0, rss0 = system_used_mb(), rss_mb()
    t_load = time.perf_counter()
    backend = make_backend(args.backend, args.engine, args.weights, W, H, args.widths)
    load_s = time.perf_counter() - t_load
    run_single(backend, imgs, 20, 0)      # make sure lazy allocations have happened
    used1, rss1 = system_used_mb(), rss_mb()

    print("Backend %s loaded in %.1f s | process RSS +%.0f MB | system used +%.0f MB"
          % (backend.name, load_s, rss1 - rss0, used1 - used0))

    pred_ms, post_ms, total_ms = run_single(backend, imgs, args.n, args.warmup)
    res = {
        "backend": backend.name,
        "input": [W, H],
        "env": env_info(),
        "load_s": load_s,
        "memory_mb": {"rss_after": rss1, "rss_delta": rss1 - rss0,
                      "system_used_after": used1, "system_used_delta": used1 - used0},
        "model": {"params": analytic_params(tuple(args.widths)),
                  "gmacs": analytic_macs(H, W, tuple(args.widths)) / 1e9,
                  "gflops": 2 * analytic_macs(H, W, tuple(args.widths)) / 1e9,
                  "widths": list(args.widths)},
        "latency_ms": {"predict": summarize(pred_ms), "post": summarize(post_ms),
                       "total": summarize(total_ms)},
        "fps_single_eye": 1000.0 / float(np.mean(total_ms)),
    }
    if args.backend == "trt" and os.path.exists(args.engine):
        res["engine_size_mb"] = os.path.getsize(args.engine) / (1024.0 * 1024.0)
    if args.backend.startswith("torch") and os.path.exists(args.weights):
        res["weights_size_mb"] = os.path.getsize(args.weights) / (1024.0 * 1024.0)

    if args.two_eyes:
        res["two_eyes"] = run_two_eyes(backend, imgs, args.n)

    out_dir = os.path.dirname(os.path.abspath(args.out))
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)

    t = res["latency_ms"]["total"]
    print("Latency total ms: mean %.2f | SD %.2f | median %.2f | p95 %.2f | p99 %.2f | max %.2f"
          % (t["mean"], t["sd"], t["median"], t["p95"], t["p99"], t["max"]))
    print("Single-eye throughput: %.1f fps" % res["fps_single_eye"])
    if args.two_eyes:
        te = res["two_eyes"]
        print("Two eyes (shared backend): %.1f fps per eye | %.1f fps combined"
              % (te["fps_per_eye"], te["fps_combined"]))
    print("Model: %.2f M params | %.2f GMACs (%.2f GFLOPs)"
          % (res["model"]["params"] / 1e6, res["model"]["gmacs"], res["model"]["gflops"]))
    print("Saved:", args.out)


if __name__ == "__main__":
    main()
