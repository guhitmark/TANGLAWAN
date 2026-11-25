"""TANGLAWAN 90-second research protocol: quality-aware PLR metrics.

Red: baseline 0–10 s; fixed light 10–12 s; response through 32 s.
Blue: baseline 47–57 s; fixed light 57–59 s; response through 90 s.

Nine returned string values, in the original API order:
 baseline (mm), minimum (mm), constriction amplitude (mm), latency (ms),
 maximum constriction velocity (mm/s), average constriction velocity (mm/s),
 PIPR 6 s (mm baseline-minus-diameter), maximum redilation velocity (mm/s),
 75% recovery (ms after minimum).

Derived metrics are exploratory, *not validated clinical biomarkers*.
Only measured, quality-accepted samples are used; no held/interpolated values.
"""
import math
import numpy as np

RED_BASELINE = (0.0, 10.0)
RED_STIM = (10.0, 12.0)
RED_END = 32.0
BLUE_BASELINE = (47.0, 57.0)
BLUE_STIM = (57.0, 59.0)
BLUE_END = 90.0
EMPTY = ("--",) * 9


def _good_number(v):
    try:
        x = float(v)
        return x if math.isfinite(x) else float('nan')
    except (ValueError, TypeError, OverflowError):
        return float('nan')


def _valid_flag(v):
    if isinstance(v, str):
        return v.strip().lower() in ('1', 'true', 'yes')
    return bool(v)


def _median_at(t, y, center, radius=0.15, min_n=2):
    m = (t >= center-radius) & (t <= center+radius)
    if m.sum() < min_n:
        return float('nan')
    return float(np.median(y[m]))


def calculate_clinical_metrics(times, diams, t_base_start, t_base_end,
                               t_stim, t_stim_off=None, t_end=None):
    """Return the existing nine-metric tuple; optional arguments allow old callers.

    Avoid fitting/smoothing across long tracking gaps. PIPR is the residual
    constriction in mm at 6 seconds AFTER light offset, measured as baseline
    minus median diameter in a 5–7 s post-offset window.
    """
    if times is None or diams is None or len(times) == 0 or len(diams) == 0:
        return EMPTY
    t = np.asarray(times, dtype=float)
    y = np.asarray(diams, dtype=float)
    if t.shape != y.shape:
        return EMPTY
    good = np.isfinite(t) & np.isfinite(y) & (y >= 1.0) & (y <= 9.0)
    t, y = t[good], y[good]
    if t.size < 5:
        return EMPTY
    order = np.argsort(t, kind='mergesort')
    t, y = t[order], y[order]
    if t_stim_off is None:
        t_stim_off = float(t_stim) + 2.0
    if t_end is None:
        t_end = float(t[-1]) + 1e-6
    # Never include the next stimulus, washout, or another phase.
    window = (t >= t_base_start) & (t < t_end)
    t, y = t[window], y[window]
    base = (t >= t_base_start) & (t < t_base_end)
    if base.sum() < 5:
        return EMPTY
    baseline = float(np.median(y[base]))

    # Light onset through the first 8s: search for early peak constriction,
    # not an arbitrary later minimum/late blink in the recovery window.
    response = (t >= t_stim) & (t <= min(t_stim + 8.0, t_end))
    if response.sum() < 4:
        return (f'{baseline:.2f}',) + ('--',)*8

    # 0.30 s moving median for extrema/velocities; do not fill missing frames.
    sm = np.array([_median_at(t, y, v, 0.15, 2) for v in t])
    sm[~np.isfinite(sm)] = y[~np.isfinite(sm)]
    idx = np.flatnonzero(response)
    i_min = idx[np.argmin(sm[idx])]
    minimum = float(sm[i_min])
    t_min = float(t[i_min])
    amp = max(0.0, baseline-minimum)

    # Require a sustained departure from baseline (at least ~150ms).
    latency = '--'
    threshold = max(0.05, 0.03 * baseline)
    candidates = np.flatnonzero((t >= t_stim) & (t <= min(t_stim+5.0, t_end)) &
                                 (sm <= baseline - threshold))
    for i in candidates:
        future = (t >= t[i]) & (t < t[i]+0.20)
        if future.sum() >= 2 and np.median(sm[future]) <= baseline-threshold:
            latency = str(int(round(1000.0*(t[i]-t_stim))))
            break

    # Central differences across close observed timestamps, avoid huge values
    # from tiny frame-to-frame noise. Segments exceeding 0.35s aren't joined.
    dt = np.diff(t)
    dy = np.diff(sm)
    valid_seg = (dt >= 0.015) & (dt <= 0.35)
    velocity = np.full(dt.shape, np.nan)
    velocity[valid_seg] = dy[valid_seg]/dt[valid_seg]
    midpoint = (t[:-1]+t[1:])/2.0
    con = (midpoint >= t_stim) & (midpoint <= t_min) & np.isfinite(velocity)
    negative = -velocity[con]
    negative = negative[negative > 0]
    max_c = f'{np.max(negative):.2f}' if negative.size else '--'
    avg_c = f'{np.mean(negative):.2f}' if negative.size else '--'

    redil = (midpoint > t_min) & (midpoint <= t_end) & np.isfinite(velocity)
    pos = velocity[redil]
    pos = pos[pos > 0]
    max_r = f'{np.max(pos):.2f}' if pos.size else '--'

    # 6s PIPR: median at 5–7 seconds after light offset.
    pipr_window = (t >= t_stim_off+5) & (t <= t_stim_off+7)
    pipr = '--'
    if pipr_window.sum() >= 3:
        pipr = f'{baseline-np.median(y[pipr_window]):.2f}'

    recovery = '--'
    target = minimum + 0.75*amp
    if amp >= 0.10:
        for i in np.flatnonzero((t > t_min) & (t <= t_end) & (sm >= target)):
            future = (t >= t[i]) & (t < t[i]+0.20)
            if future.sum() >= 2 and np.median(sm[future]) >= target:
                recovery = str(int(round(1000.0*(t[i]-t_min))))
                break

    return (f'{baseline:.2f}', f'{minimum:.2f}', f'{amp:.2f}', latency,
            max_c, avg_c, pipr, max_r, recovery)


def get_clinical_report(session_data_eye):
    """Receive one eye's session rows with 't', 'y', 'valid'."""
    red_t, red_y, blue_t, blue_y = [], [], [], []
    for row in session_data_eye:
        if not isinstance(row, dict) or not _valid_flag(row.get('valid', False)):
            continue
        # Exclude held/reconstructed values if quality fields are supplied.
        if _valid_flag(row.get('filter_held', False)) or _valid_flag(row.get('blink_repaired', False)):
            continue
        t, y = _good_number(row.get('t')), _good_number(row.get('y'))
        if not (math.isfinite(t) and math.isfinite(y)):
            continue
        if RED_BASELINE[0] <= t <= RED_END:
            red_t.append(t); red_y.append(y)
        elif BLUE_BASELINE[0] <= t <= BLUE_END:
            blue_t.append(t); blue_y.append(y)

    red = calculate_clinical_metrics(red_t, red_y, *RED_BASELINE,
                                     RED_STIM[0], RED_STIM[1], RED_END)
    blue = calculate_clinical_metrics(blue_t, blue_y, *BLUE_BASELINE,
                                      BLUE_STIM[0], BLUE_STIM[1], BLUE_END)
    return red, blue
