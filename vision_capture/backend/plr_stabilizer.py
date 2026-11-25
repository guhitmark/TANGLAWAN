"""Phase-aware PLR display/export stabilization; never changes raw observations.

Elapsed times (seconds): red pulse 10–12, blue pulse 57–59.
Held values are visualization estimates, NOT measured diameters.
V4 rejects narrow returning artifacts even inside protected windows.
Short gaps are bridged for display only; CSV holds remain explicit.
Thresholds are engineering starting points, not clinically validated.
Python 3.6+ / NumPy only for signal processing; stdlib for CSV export.
"""
import csv
import math
import os
import tempfile
import numpy as np

MIN_CONFIDENCE = 0.5
MIN_MM, MAX_MM = 1.0, 9.0
PROTECTED = ((9.75, 15.3), (56.75, 62.3))
BLINK_PAD_S = 0.12
# Engineering thresholds; verify against raw eye video.
TRANSIENT_MM = 0.11
TRANSIENT_EDGE_MM = 0.035
MAX_TRANSIENT_S = 2.2
SHOULDER_RADIUS_S = 0.13
SHOULDER_OFFSETS_S = (0.18, 0.30, 0.45, 0.65, 0.85, 1.10)
SHOULDER_AGREEMENT_MM = 0.20
# Robust two-stage smoothing, outside the protected stimulus response.
SMOOTH_RADIUS_S = 0.43
GAUSSIAN_SIGMA_S = 0.27
GAUSSIAN_RADIUS_S = 0.81
PROTECTION_BLEND_S = 0.40
# Display-only marker lane, visible in the current PNG/HTML/report axes.
MARKER_LANE_MM = 1.8
MAX_SAMPLE_GAP_S = 0.45
FILTER_VERSION = 'plr_stabilizer_90s_v5'
# Conservative narrow-spike detector (also active during the trigger).
TRIGGER_SPIKE_MM = 0.40
CORE_SPIKE_MM = 0.75
TRIGGER_SPIKE_EDGE_MM = 0.14
TRIGGER_SPIKE_MAX_S = 0.42
# Display-only: bridge short gaps, never extrapolate or bridge longer gaps.
DISPLAY_BRIDGE_MAX_S = 2.6
DISPLAY_SIGMA_S = 0.40
DISPLAY_TRIGGER_SIGMA_S = 0.10
DISPLAY_BLEND_S = 0.6


def number(value):
    try:
        x = float(value)
        return x if math.isfinite(x) else float('nan')
    except (TypeError, ValueError, OverflowError):
        return float('nan')


def flag(value):
    return str(value).strip().lower() in ('1', '1.0', 'true', 'yes')


def protected(t):
    return any(start <= t <= end for start, end in PROTECTED)


def phase(t):
    if 0 <= t <= 32:
        return 'red'
    if 47 <= t <= 90:
        return 'blue'
    return None


def _returning_excursions(t, y, accepted, protect):
    """Offline detector: reject short returning excursions outside trigger windows.
    Use observed shoulders only; never search across a real sampling gap.
    """
    rejected = np.zeros(len(t), dtype=bool)
    start = 0
    while start < len(t):
        if protect[start]:
            start += 1
            continue
        end = start + 1
        while (end < len(t) and not protect[end] and
               t[end] - t[end-1] <= MAX_SAMPLE_GAP_S):
            end += 1
        ts, ys, ok = t[start:end], y[start:end], accepted[start:end]
        n = len(ts)
        if n < 5:
            start = end
            continue
        for span in SHOULDER_OFFSETS_S:
            width = min(SHOULDER_RADIUS_S, span * 0.65)
            left_lo = np.searchsorted(ts, ts-span-width, side='left')
            left_hi = np.searchsorted(ts, ts-span+width, side='right')
            right_lo = np.searchsorted(ts, ts+span-width, side='left')
            right_hi = np.searchsorted(ts, ts+span+width, side='right')
            up = np.zeros(n); down = np.zeros(n)
            for i in np.flatnonzero(ok):
                if ts[i]-span-width < ts[0] or ts[i]+span+width > ts[-1]:
                    continue
                a,b,c,d = left_lo[i],left_hi[i],right_lo[i],right_hi[i]
                lv, rv = ys[a:b][ok[a:b]], ys[c:d][ok[c:d]]
                if (len(lv) < max(2, (b-a+1)//2) or
                    len(rv) < max(2, (d-c+1)//2)):
                    continue
                l, r = np.median(lv), np.median(rv)
                if abs(l-r) > SHOULDER_AGREEMENT_MM + 0.04*span:
                    continue
                down[i] = min(l,r)-ys[i]
                up[i] = ys[i]-max(l,r)
            for deviation in (down, up):
                loose = ok & (deviation >= TRANSIENT_EDGE_MM)
                edges = np.flatnonzero(np.diff(np.r_[False, loose, False]))
                for a,b in zip(edges[::2], edges[1::2]):
                    if (ts[b-1]-ts[a] <= MAX_TRANSIENT_S and
                        np.max(deviation[a:b]) >= TRANSIENT_MM):
                        rejected[start+a:start+b] = True
        start = end
    # Prevent overlapping candidates from creating an unlimited artifact hold.
    edges = np.flatnonzero(np.diff(np.r_[False, rejected, False]))
    for a,b in zip(edges[::2], edges[1::2]):
        if t[b-1]-t[a] > MAX_TRANSIENT_S:
            rejected[a:b] = False
    return rejected


def _trigger_spikes(t, y, accepted, protect):
    """Find narrow, deep returning excursions, not the broad PLR trough.

    Both observed shoulders must support a return. In the early response
    use a higher threshold and shorter allowed duration. Missing samples
    are never used as evidence. Thresholds require device-specific review.
    """
    reject = np.zeros(len(t), dtype=bool)
    for span in (0.12, 0.19, 0.27, 0.36):
        width = min(0.105, 0.58 * span)
        a = np.searchsorted(t, t-span-width, 'left')
        b = np.searchsorted(t, t-span+width, 'right')
        c = np.searchsorted(t, t+span-width, 'left')
        d = np.searchsorted(t, t+span+width, 'right')
        dev = np.zeros(len(t))
        for i in np.flatnonzero(accepted & protect):
            li = np.arange(a[i], b[i]); ri = np.arange(c[i], d[i])
            li = li[accepted[li]]; ri = ri[accepted[ri]]
            if len(li) < 2 or len(ri) < 2:
                continue
            if np.max(np.diff(t[li[0]:ri[-1]+1])) > MAX_SAMPLE_GAP_S:
                continue
            l, r = np.median(y[li]), np.median(y[ri])
            # An actual steep monotonic constriction/recovery cannot
            # be below BOTH shoulders (or above both).
            dev[i] = max(min(l,r)-y[i], y[i]-max(l,r), 0.0)
        edges = np.flatnonzero(np.diff(np.r_[False, dev >= TRIGGER_SPIKE_EDGE_MM, False]))
        for st,en in zip(edges[::2],edges[1::2]):
            core = any(onset <= t[st] <= onset + 2.1 for onset in (10.,57.))
            threshold = CORE_SPIKE_MM if core else TRIGGER_SPIKE_MM
            duration = 0.23 if core else TRIGGER_SPIKE_MAX_S
            if t[en-1]-t[st] <= duration and np.max(dev[st:en]) >= threshold:
                reject[st:en] = True
    return reject


def _display_curve(t, y, valid):
    """Offline visual trend only. Fit accepted samples, not held plateaus.

    A local Gaussian linear fit smooths valid samples with a short kernel
    around stimulus onset. Interpolate short, bounded quality gaps with
    a slope-limited cubic Hermite curve (no overshoot); never bridge a
    long loss or a real timestamp discontinuity. Raw/filtered CSV values
    and flags are untouched.
    """
    result = np.full(len(t), np.nan)
    ix = np.flatnonzero(valid & np.isfinite(y))
    if not len(ix):
        return result
    # Split wherever the acquisition itself stops, or too much is missing.
    split = []
    for j in range(1,len(ix)):
        left,right = ix[j-1],ix[j]
        if (t[right]-t[left] > DISPLAY_BRIDGE_MAX_S or
            np.any(np.diff(t[left:right+1]) > MAX_SAMPLE_GAP_S)):
            split.append(j)
    for group in np.split(ix, split):
        ts,ys = t[group], y[group]
        sm = ys.copy()
        for j,x in enumerate(ts):
            # Smoothly vary bandwidth, rather than switch filters abruptly.
            distance = min((max(a-x, x-b, 0.) for a,b in PROTECTED))
            u = min(1., distance/DISPLAY_BLEND_S)
            u = u*u*(3.-2.*u)
            sigma = DISPLAY_TRIGGER_SIGMA_S + u*(DISPLAY_SIGMA_S-DISPLAY_TRIGGER_SIGMA_S)
            lo = np.searchsorted(ts,x-3*sigma,'left')
            hi = np.searchsorted(ts,x+3*sigma,'right')
            dx = ts[lo:hi]-x; v = ys[lo:hi]
            w = np.exp(-0.5*(dx/sigma)**2)
            sw = np.sum(w); sx = np.dot(w,dx); sxx=np.dot(w,dx*dx)
            sy=np.dot(w,v); sxy=np.dot(w,dx*v); det=sw*sxx-sx*sx
            sm[j] = np.clip((sxx*sy-sx*sxy)/det if det>1e-10 else sy/sw, np.min(v), np.max(v))
        result[group] = sm
        # Monotone cubic interpolation. Use adjacent measured slopes
        # when available, limiting them to prevent artificial extrema.
        for j in range(len(group)-1):
            l,r = group[j],group[j+1]
            if r <= l+1 or t[r] <= t[l]:
                continue
            h = t[r]-t[l]; v0,v1 = sm[j],sm[j+1]; slope=(v1-v0)/h
            m0 = ((v0-sm[j-1])/(ts[j]-ts[j-1])
                  if j>0 and ts[j]>ts[j-1] else slope)
            m1 = ((sm[j+2]-v1)/(ts[j+2]-ts[j+1])
                  if j+2<len(group) and ts[j+2]>ts[j+1] else slope)
            if abs(slope)<1e-10:
                m0=m1=0.
            else:
                a=max(0.,m0/slope); b=max(0.,m1/slope)
                scale=max(1., math.hypot(a,b)/3.)
                m0=a*slope/scale; m1=b*slope/scale
            u=(t[l+1:r]-t[l])/h
            result[l+1:r] = ((2*u**3-3*u**2+1)*v0 + (u**3-2*u**2+u)*h*m0
                             + (-2*u**3+3*u**2)*v1 + (u**3-u**2)*h*m1)
    return result


def _phase_filter(rows):
    """Input sorted by time. Output aligned rows, with independent quality flags."""
    if not rows:
        return []
    t = np.array([number(r['t']) for r in rows])
    y = np.array([number(r.get('y')) for r in rows])
    protect = np.array([protected(x) for x in t], dtype=bool)
    good = np.array([flag(r.get('valid', False)) and
                     (r.get('confidence') is None or number(r['confidence']) >= MIN_CONFIDENCE)
                     for r in rows], dtype=bool)
    good &= np.isfinite(y) & (y >= MIN_MM) & (y <= MAX_MM)
    good &= np.array([not flag(r.get('blink_repaired', False)) for r in rows], dtype=bool)
    reason = np.where(good, 'measured', 'tracking_lost').astype(object)
    accepted = good.copy()
    # Eyelid edges often remain marked valid just before/after a blink.
    # Do not pad in the protected trigger interval.
    for i in np.flatnonzero(~good):
        lo = np.searchsorted(t, t[i] - BLINK_PAD_S, side='left')
        hi = np.searchsorted(t, t[i] + BLINK_PAD_S, side='right')
        ix = np.arange(lo, hi)
        ix = ix[good[ix] & ~protect[ix]]
        accepted[ix] = False
        reason[ix] = 'near_tracking_loss'

    # Detect rounded blink-like troughs using both surrounding shoulders.
    artifact = _returning_excursions(t, y, accepted, protect)
    accepted[artifact] = False
    reason[artifact] = 'returning_transient'
    trigger_artifact = _trigger_spikes(t, y, accepted, protect)
    accepted[trigger_artifact] = False
    reason[trigger_artifact] = 'narrow_trigger_artifact'

    # Median followed by time-weighted Gaussian smoothing. Both stages
    # operate only on observed/accepted contiguous samples; they never
    # span a blink, missing interval or protected stimulus boundary.
    smooth = y.copy()
    start = 0
    while start < len(t):
        if not accepted[start] or protect[start]:
            start += 1
            continue
        end = start + 1
        while (end < len(t) and accepted[end] and not protect[end]
               and t[end]-t[end-1] <= MAX_SAMPLE_GAP_S):
            end += 1
        ts, ys = t[start:end], y[start:end]
        med = np.empty(len(ts), dtype=float)
        lo = np.searchsorted(ts, ts-SMOOTH_RADIUS_S, side='left')
        hi = np.searchsorted(ts, ts+SMOOTH_RADIUS_S, side='right')
        for i, (a, b) in enumerate(zip(lo, hi)):
            med[i] = np.median(ys[a:b])
        lo = np.searchsorted(ts, ts-GAUSSIAN_RADIUS_S, side='left')
        hi = np.searchsorted(ts, ts+GAUSSIAN_RADIUS_S, side='right')
        # Local linear Gaussian fit mitigates boundary bias on slopes.
        # Bound the result by nearby median values to avoid overshoot.
        for i, (a, b) in enumerate(zip(lo, hi)):
            dx = ts[a:b] - ts[i]
            w = np.exp(-0.5 * (dx / GAUSSIAN_SIGMA_S)**2)
            v = med[a:b]
            sw = np.sum(w)
            sx = np.dot(w, dx)
            sxx = np.dot(w, dx*dx)
            sy = np.dot(w, v)
            sxy = np.dot(w, dx*v)
            det = sw*sxx - sx*sx
            filtered = ((sxx*sy-sx*sxy)/det if det > 1e-10 else sy/sw)
            filtered = np.clip(filtered, np.min(v), np.max(v))
            # Feather into/out of protected windows so a filter change
            # does not introduce an artificial jump at the boundary.
            distance = min(min(abs(ts[i]-a0), abs(ts[i]-b0))
                           for a0,b0 in PROTECTED)
            u = min(1.0, distance / PROTECTION_BLEND_S)
            blend = u*u*(3.0-2.0*u)
            smooth[start+i] = (1.0-blend)*ys[i] + blend*filtered
        start = end

    result, last, last_t = [], None, None
    for i, r in enumerate(rows):
        if accepted[i]:
            last = float(smooth[i]); last_t = float(t[i])
        row = dict(r)
        row.update(raw_y=r.get('y'), filtered_mm=last,
                   filter_valid=bool(accepted[i]),
                   filter_held=bool(not accepted[i] and last is not None),
                   filter_reason=str(reason[i]),
                   hold_age_s=(float(t[i])-last_t if not accepted[i] and last_t is not None else None),
                   filter_version=FILTER_VERSION)
        result.append(row)
    return result


def stabilize_session(rows):
    """Return row copies in original order. Never backfill before first valid sample.
    The dark/readaptation period and beyond-protocol rows are not estimated.
    """
    result = []
    buckets = {'red': [], 'blue': []}
    for i, r in enumerate(rows):
        p = phase(number(r.get('t')))
        if p:
            item = dict(r); item['_stabilizer_index'] = i
            buckets[p].append(item)
        result.append(dict(r, raw_y=r.get('y'), filtered_mm=None,
                           filter_valid=False, filter_held=False,
                           filter_reason='outside_protocol', hold_age_s=None,
                           filter_version=FILTER_VERSION))
    for group in buckets.values():
        group.sort(key=lambda r: number(r['t']))
        for row in _phase_filter(group):
            idx = row.pop('_stabilizer_index')
            result[idx] = row
    return result


def plot_series(rows):
    """Continuous display + a separate quality-marker lane.

    Backwards compatibility with existing exporters.py/report_generator.py:
    ``measured`` is the smoothed display curve, with short bounded gaps
    interpolated; it is not a sequence of independent measurements; ``held`` is an empty legacy
    dashed overlay, so older exporters draw only a solid curve.
    ``held_y`` is a display marker coordinate, NOT pupil diameter.
    ``held_value_y`` and ``accepted_only`` expose the original meanings
    for callers that require them. CSV flags remain authoritative.
    Actual timestamp gaps, long losses and leading/trailing missing values remain gaps.
    CSV held values are preserved separately.
    """
    out = {}
    for name in ('red', 'blue'):
        group = sorted([r for r in rows if phase(number(r.get('t'))) == name],
                       key=lambda r: number(r['t']))
        t = np.array([number(r['t']) for r in group], dtype=float)
        y = np.array([number(r['filtered_mm']) for r in group], dtype=float)
        valid = np.array([r['filter_valid'] for r in group], dtype=bool)
        held = np.array([r['filter_held'] for r in group], dtype=bool)
        visible = _display_curve(t, y, valid)
        gap = np.flatnonzero(np.diff(t) > MAX_SAMPLE_GAP_S) + 1
        curve = np.insert(visible, gap, np.nan)
        out[name] = dict(t=np.insert(t, gap, np.nan),
                         measured=curve,
                         held=np.full(curve.shape, np.nan),
                         continuous=curve,
                         accepted_only=np.insert(np.where(valid, y, np.nan), gap, np.nan),
                         display_interpolated=np.insert((~valid)&np.isfinite(visible), gap, False),
                         held_t=t[held],
                         held_y=np.full(np.count_nonzero(held), MARKER_LANE_MM),
                         held_value_y=y[held],
                         missing_t=t[~valid & ~held])
    return out


CSV_FIELDS = ['filtered_mm', 'filtered_px', 'filter_valid', 'filter_held',
              'filter_reason', 'hold_age_s', 'filter_version']


def add_filtered_csv(path, output_path=None):
    """Append derived columns; original column values stay unchanged.
    Atomic replacement when output_path is omitted; safe to rerun.
    """
    with open(path, 'r', newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        columns = list(reader.fieldnames or [])
        rows = list(reader)
    if not {'timestamp_s', 'valid'}.issubset(columns) or not (
            'smoothed_mm' in columns or 'diameter_mm' in columns):
        raise ValueError('Missing timestamp_s, valid or diameter columns: '+str(path))
    source = 'smoothed_mm' if 'smoothed_mm' in columns else 'diameter_mm'
    session = [dict(t=r['timestamp_s'], y=r[source], valid=r['valid'],
                    confidence=r.get('confidence'), blink_repaired=r.get('blink_repaired', 0)) for r in rows]
    cleaned = stabilize_session(session)
    # Estimate existing calibration, not a newly invented constant.
    ratios = []
    for r in rows:
        mm, px = number(r.get('diameter_mm')), number(r.get('diameter_px'))
        if flag(r['valid']) and mm > 0 and px > 0:
            ratios.append(px/mm)
    ppm = float(np.median(ratios)) if ratios else None
    def fmt(x):
        return '' if x is None else '{:.6f}'.format(x)
    for r, q in zip(rows, cleaned):
        mm = q['filtered_mm']
        r.update(filtered_mm=fmt(mm), filtered_px=fmt(mm*ppm if mm is not None and ppm is not None else None),
                 filter_valid=int(q['filter_valid']), filter_held=int(q['filter_held']),
                 filter_reason=q['filter_reason'], hold_age_s=fmt(q['hold_age_s']),
                 filter_version=FILTER_VERSION)
    target = os.path.abspath(output_path or path)
    fd, tmp = tempfile.mkstemp(prefix='.plr_', suffix='.csv', dir=os.path.dirname(target))
    try:
        with os.fdopen(fd, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=columns+[k for k in CSV_FIELDS if k not in columns])
            writer.writeheader(); writer.writerows(rows)
        os.replace(tmp, target)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)
    return target


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Append quality-marked PLR filtered values to CSV.')
    parser.add_argument('source'); parser.add_argument('output')
    args = parser.parse_args()
    print(add_filtered_csv(args.source, args.output))
