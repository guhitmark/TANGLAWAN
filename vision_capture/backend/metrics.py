"""
metrics.py
---------------------------------------------------------------------
NEW FILE. Clinical PLR metrics, ported verbatim (math only) from the
desktop app's video_panel.py -- the math itself never touched
Tkinter, so this is a straight lift into a pure-function module.
---------------------------------------------------------------------
"""

import math
import numpy as np


def calculate_clinical_metrics(times, diams, t_base_start, t_base_end, t_stim):
    if not times or not diams:
        return ("--",) * 9

    t_arr = np.array(times)
    y_arr = np.array(diams)

    valid_idx = ~np.isnan(y_arr)
    t_arr = t_arr[valid_idx]
    y_arr = y_arr[valid_idx]
    if len(y_arr) == 0:
        return ("--",) * 9

    base_mask = (t_arr >= t_base_start) & (t_arr < t_base_end)
    if not np.any(base_mask):
        return ("--",) * 9
    base_diam = np.nanmean(y_arr[base_mask])

    post_mask = (t_arr >= t_stim)
    if not np.any(post_mask):
        return (f"{base_diam:.2f}",) + ("--",) * 8

    min_diam = np.nanmin(y_arr[post_mask])
    min_idx = np.nanargmin(y_arr[post_mask])
    t_min = t_arr[post_mask][min_idx]
    amp = base_diam - min_diam

    latency = "--"
    for i, t in enumerate(t_arr[post_mask]):
        if y_arr[post_mask][i] < (base_diam - 0.05):
            latency = f"{int((t - t_stim) * 1000)}"
            break

    dy = np.diff(y_arr)
    dt = np.diff(t_arr)
    dt[dt == 0] = 0.001
    velocities = dy / dt

    constrict_mask = (t_arr[:-1] >= t_stim) & (t_arr[:-1] <= t_min)
    if np.any(constrict_mask):
        avg_c_vel = f"{abs(np.nanmean(velocities[constrict_mask])):.2f}"
        max_c_vel = f"{abs(np.nanmin(velocities[constrict_mask])):.2f}"
    else:
        avg_c_vel, max_c_vel = "--", "--"

    redil_mask = (t_arr[:-1] > t_min)
    max_r_vel = f"{np.nanmax(velocities[redil_mask]):.2f}" if np.any(redil_mask) else "--"

    pipr_mask = (t_arr >= t_arr[-1] - 10)
    pipr = f"{np.nanmean(y_arr[pipr_mask]):.2f}" if np.any(pipr_mask) else "--"

    rec_75 = "--"
    target_rec = min_diam + (0.75 * amp)
    if np.any(redil_mask):
        for i, t in enumerate(t_arr[:-1][redil_mask]):
            if y_arr[:-1][redil_mask][i] >= target_rec:
                rec_75 = f"{int((t - t_min) * 1000)}"
                break

    return (f"{base_diam:.2f}", f"{min_diam:.2f}", f"{amp:.2f}", latency,
            max_c_vel, avg_c_vel, pipr, max_r_vel, rec_75)


def get_clinical_report(session_data_eye):
    """session_data_eye: list of {'t':.., 'y':.., 'valid':..} for ONE eye."""
    t_r_v, y_r_v = [], []
    t_b_v, y_b_v = [], []

    for r in session_data_eye:
        if not r["valid"] or math.isnan(r["y"]):
            continue
        if r["t"] <= 60:
            t_r_v.append(r["t"]); y_r_v.append(r["y"])
        elif 240 <= r["t"] <= 300:
            t_b_v.append(r["t"]); y_b_v.append(r["y"])

    r_met = calculate_clinical_metrics(t_r_v, y_r_v, 0, 10, 10)
    b_met = calculate_clinical_metrics(t_b_v, y_b_v, 240, 250, 250)
    return r_met, b_met