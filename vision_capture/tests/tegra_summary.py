"""tegra_summary.py - summarize a tegrastats log (RAM, swap, CPU, GPU load, temperature, power).

  sudo tegrastats --interval 500 --logfile results/tegra_trt.log --start
  ... run your workload ...
  sudo tegrastats --stop
  python3 tegra_summary.py results/tegra_trt.log --trim 4 > results/tegra_trt.json

--trim N drops the first and last N samples (start-up / shut-down).
Power is POM_5V_IN in mW (whole module input) when the board reports it.
"""
import argparse
import json
import re
import sys

import numpy as np

RAM = re.compile(r"RAM (\d+)/(\d+)MB")
SWAP = re.compile(r"SWAP (\d+)/(\d+)MB")
CPU = re.compile(r"CPU \[([^\]]*)\]")
GR3D = re.compile(r"GR3D_FREQ (\d+)%")
T_CPU = re.compile(r"CPU@([\d.]+)C")
T_GPU = re.compile(r"GPU@([\d.]+)C")
T_TH = re.compile(r"thermal@([\d.]+)C")
PWR = re.compile(r"POM_5V_IN (\d+)/(\d+)")
CORE = re.compile(r"(\d+)%")


def stat(vals):
    a = np.asarray(vals, dtype=np.float64)
    if a.size == 0:
        return None
    return {"mean": float(a.mean()), "sd": float(a.std(ddof=1)) if a.size > 1 else 0.0,
            "p95": float(np.percentile(a, 95)), "max": float(a.max())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logfile")
    ap.add_argument("--trim", type=int, default=4)
    args = ap.parse_args()

    samples = []
    with open(args.logfile) as f:
        for line in f:
            m = RAM.search(line)
            if not m:
                continue
            s = {"ram_used_mb": int(m.group(1)), "ram_total_mb": int(m.group(2))}
            m = SWAP.search(line)
            if m:
                s["swap_used_mb"] = int(m.group(1))
            m = CPU.search(line)
            if m:
                loads = [int(x) for x in CORE.findall(m.group(1))]
                if loads:
                    s["cpu_avg_pct"] = float(np.mean(loads))
                    s["cpu_max_core_pct"] = float(max(loads))
            m = GR3D.search(line)
            if m:
                s["gpu_load_pct"] = int(m.group(1))
            for key, rx in (("temp_cpu_c", T_CPU), ("temp_gpu_c", T_GPU), ("temp_thermal_c", T_TH)):
                m = rx.search(line)
                if m:
                    s[key] = float(m.group(1))
            m = PWR.search(line)
            if m:
                s["power_in_mw"] = int(m.group(1))
            samples.append(s)

    if not samples:
        sys.exit("No tegrastats samples parsed from %s (empty log?)" % args.logfile)
    if args.trim and len(samples) > 2 * args.trim + 5:
        samples = samples[args.trim:-args.trim]

    out = {"samples": len(samples)}
    keys = set()
    for s in samples:
        keys.update(s.keys())
    for k in sorted(keys):
        if k == "ram_total_mb":
            out[k] = samples[0][k]
            continue
        out[k] = stat([s[k] for s in samples if k in s])
    if "ram_used_mb" in out and out.get("ram_total_mb"):
        out["ram_used_pct_mean"] = 100.0 * out["ram_used_mb"]["mean"] / out["ram_total_mb"]
    json.dump(out, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
