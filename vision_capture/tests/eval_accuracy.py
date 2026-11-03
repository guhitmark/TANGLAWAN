"""eval_accuracy.py - per-image accuracy of a backend on the held-out TEST split.

Run from the project root (next to run.py), on the Jetson:

  python3 eval_accuracy.py --test-dir ~/data/annotated/test --backend trt \
          --engine controllers/pupil_tracker_fp16.engine --out results/acc_trt.csv
  python3 eval_accuracy.py --test-dir ~/data/annotated/test --backend torch32 \
          --weights controllers/pupil_tracker_best.pth --out results/acc_torch32.csv

--backend dummy is a pipeline self-test (not a real model).
Writes one CSV row per image; analyze_results.py turns the CSVs into tables/figures.
"""
import argparse
import csv
import os
import sys
import time

import cv2
import numpy as np

from backends import make_backend, seg_metrics, mask_to_ellipse, list_pairs, DEFAULT_WIDTHS

FIELDS = ["image", "dice", "iou", "precision", "recall", "pixel_acc", "gt_area", "pred_area",
          "gt_found", "pred_found", "gt_cx", "gt_cy", "gt_diam_px",
          "pred_cx", "pred_cy", "pred_diam_px", "diam_err_px", "center_err_px"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-dir", required=True, help="folder with images/ and masks/ (the TEST split)")
    ap.add_argument("--backend", required=True, choices=["trt", "torch32", "torch16", "dummy"])
    ap.add_argument("--engine", default="controllers/pupil_lite_400x400_fp16.engine")
    ap.add_argument("--weights", default="controllers/pupil_lite_best.pth")
    ap.add_argument("--widths", type=int, nargs=5, default=list(DEFAULT_WIDTHS))
    ap.add_argument("--size", type=int, nargs=2, default=[400, 400], metavar=("W", "H"))
    ap.add_argument("--limit", type=int, default=0, help="only the first N images (0 = all)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    W, H = args.size
    pairs = list_pairs(args.test_dir)
    if args.limit:
        pairs = pairs[:args.limit]
    if not pairs:
        sys.exit("No image/mask pairs found in %s (expected images/X.jpg and masks/X_mask.png)" % args.test_dir)

    backend = make_backend(args.backend, args.engine, args.weights, W, H, args.widths)
    print("Backend: %s | %d test images | input %dx%d" % (backend.name, len(pairs), W, H))

    out_dir = os.path.dirname(os.path.abspath(args.out))
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    t_start = time.time()
    with open(args.out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=FIELDS)
        wr.writeheader()
        for i, (name, ip, mp) in enumerate(pairs):
            img = cv2.imread(ip, cv2.IMREAD_GRAYSCALE)
            gt = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
            if img is None or gt is None:
                print("skip unreadable:", name)
                continue
            if img.shape != (H, W):
                sys.exit("%s is %dx%d but the model input is %dx%d. Use --size or check the dataset."
                         % (name, img.shape[1], img.shape[0], W, H))
            gt = (gt > 127).astype(np.uint8)

            pred = backend.predict(img)
            row = {"image": name}
            row.update(seg_metrics(pred, gt))

            ge = mask_to_ellipse(gt)
            pe = mask_to_ellipse(pred)
            row["gt_found"] = int(ge is not None)
            row["pred_found"] = int(pe is not None)
            nan = float("nan")
            row["gt_cx"], row["gt_cy"], row["gt_diam_px"] = (ge[0], ge[1], ge[5]) if ge else (nan, nan, nan)
            row["pred_cx"], row["pred_cy"], row["pred_diam_px"] = (pe[0], pe[1], pe[5]) if pe else (nan, nan, nan)
            if ge and pe:
                row["diam_err_px"] = pe[5] - ge[5]
                row["center_err_px"] = float(np.hypot(pe[0] - ge[0], pe[1] - ge[1]))
            else:
                row["diam_err_px"] = nan
                row["center_err_px"] = nan
            wr.writerow(row)

            if (i + 1) % 100 == 0:
                print("  %d / %d" % (i + 1, len(pairs)))

    # quick console summary
    data = np.genfromtxt(args.out, delimiter=",", names=True)
    data = np.atleast_1d(data)
    print("\n=== %s on %d images (%.0f s) ===" % (backend.name, data.size, time.time() - t_start))
    print("Dice       mean %.4f | median %.4f" % (np.nanmean(data["dice"]), np.nanmedian(data["dice"])))
    print("IoU        mean %.4f | median %.4f" % (np.nanmean(data["iou"]), np.nanmedian(data["iou"])))
    print("Precision  mean %.4f | Recall mean %.4f" % (np.nanmean(data["precision"]), np.nanmean(data["recall"])))
    err = data["diam_err_px"]
    print("Diameter error (px): MAE %.3f | bias %+.3f | RMSE %.3f"
          % (np.nanmean(np.abs(err)), np.nanmean(err), np.sqrt(np.nanmean(err ** 2))))
    gt_n = int(np.nansum(data["gt_found"]))
    both = int(np.sum(~np.isnan(err)))
    print("Pupil detected on %d of %d frames that have one" % (both, gt_n))
    print("Saved:", args.out)


if __name__ == "__main__":
    main()
