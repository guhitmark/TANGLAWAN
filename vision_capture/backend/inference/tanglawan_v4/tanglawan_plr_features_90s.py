"""TANGLAWAN V4 90-second: shared training/deployment preprocessing (no learned parameters).
Research prototype only. Call prepare_pair() then impute_biomarkers() with the
median vector exported by the training notebook. Never interpret held samples
as measured physiology. Input timestamps are elapsed seconds from recording start.
"""
from pathlib import Path
import re
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

CLASS_TO_INDEX = {'LBL_0': 0, 'LBL_1': 1, 'LBL_2': 2}
CLASS_NAMES = {'LBL_0': 'Healthy', 'LBL_1': 'Diabetic-DR', 'LBL_2': 'Diabetic-NoDR'}
FS = 20
DT = 1 / FS
DURATION = 90
TIME = np.arange(DURATION * FS, dtype=np.float64) / FS
ONSETS = (10., 57.)
PULSE_S = 2.
CNN_PRE_S = 2
CNN_POST_S = 16
CNN_LEN = (CNN_PRE_S + CNN_POST_S) * FS
CNN_CHANNELS = ['LE_relative', 'RE_relative', 'LE_velocity', 'RE_velocity']
PX_PER_MM = 7.07714
MAX_NEAREST_GAP = .14
MAX_MEASUREMENT_GAP = .25
MIN_BASELINE_COUNT = 40
MIN_EARLY_COVERAGE = .70
MAX_CNN_GAP_S = 1.5
CONFIDENCE_MIN = .5
FEATURES = ['baseline_mm','peak_constriction_mm','peak_constriction_pct',
            'latency_s','time_to_peak_s','max_constriction_mm_s','max_dilation_mm_s',
            'recovery75_s','pipr6_pct','pipr10_pct','pipr20_pct','pipr30_pct',
            'pipr_auc_30_pct_s']
ASYMMETRY = ['baseline_mm','peak_constriction_pct','latency_s','pipr6_pct','pipr20_pct']
FEATURE_NAMES = [f'{phase}_mean_{f}' for phase in ('red','blue') for f in FEATURES] + [
    f'{phase}_asym_{f}' for phase in ('red','blue') for f in ASYMMETRY] + [
    'blue_red_peak_ratio','blue_minus_red_pipr6_pct','blue_minus_red_pipr20_pct']


def _column(df, names):
    for name in names:
        if name in df: return name
    raise ValueError(f'Missing columns: {names}')


def _num(df, name):
    return np.asarray(pd.to_numeric(df[name], errors='coerce').values, dtype=np.float64)


def _flag(df, name):
    return np.asarray(df[name].astype(str).str.strip().str.lower().isin(['1','1.0','true','yes']).values, dtype=bool)


def _runs(mask):
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False].astype(np.int8)))
    return list(zip(edges[::2], edges[1::2]))


def _segment_smooth(y, sigma):
    out = y.copy()
    for a,b in _runs(np.isfinite(y)):
        if b-a >= 3:
            out[a:b] = gaussian_filter1d(y[a:b], sigma=sigma, mode='nearest')
    return out


def resample_eye(df):
    """Preserve recording clock; reject invalid and held measurements before resampling."""
    t = _num(df, _column(df, ['timestamp_s','timestamp','time_s']))
    has_filter = 'filtered_mm' in df and 'filter_valid' in df
    dc = 'filtered_mm' if has_filter else _column(df, ['smoothed_mm','diameter_mm','smoothed_px','diameter_px'])
    d = _num(df, dc)
    if dc.endswith('_px'): d = d / PX_PER_MM
    # Do not reset to first valid frame: this would move the stimulus onset.
    if not np.isfinite(t).any(): raise ValueError('No finite timestamps')
    if np.nanmin(t) < -0.5 or np.nanmin(t) > 2.0:
        raise ValueError('Expected elapsed timestamps starting near 0, not wall-clock time')
    v = np.isfinite(t) & np.isfinite(d) & (d >= 1.) & (d <= 9.)
    if 'valid' in df: v &= _flag(df,'valid')
    if 'confidence' in df: v &= _num(df,'confidence') >= CONFIDENCE_MIN
    if has_filter: v &= _flag(df,'filter_valid')
    if 'filter_held' in df: v &= ~_flag(df,'filter_held')
    if 'blink_repaired' in df: v &= ~_flag(df,'blink_repaired')
    # Keep invalid source timestamps as barriers: do not bridge their samples
    # for biomarker calculations, even when a loss is brief.
    # Hard fail if an old 180-second recording is accidentally used.
    if np.nanmax(t) > DURATION + .25 or np.nanmax(t) < DURATION - 1.0:
        raise ValueError(f'Wrong protocol duration: requires approximately {DURATION}s recording ending at 89.95s, received {np.nanmax(t):.2f}s')
    finite_t = np.isfinite(t) & (t >= 0) & (t < DURATION)
    t, d, v = t[finite_t], d[finite_t], v[finite_t]
    order = np.argsort(t, kind='mergesort'); t,d,v = t[order],d[order],v[order]
    t, iu = np.unique(t, return_index=True); d,v = d[iu],v[iu]
    if len(t) < 2 or v.sum() < 30: raise ValueError('Insufficient measurements')
    ix = np.searchsorted(t, TIME, side='left')
    lo = np.clip(ix-1,0,len(t)-1); hi = np.clip(ix,0,len(t)-1)
    nearest = np.where(np.abs(t[lo]-TIME) < np.abs(t[hi]-TIME),lo,hi)
    exact = (np.abs(t[nearest]-TIME) < 1e-6) & v[nearest]
    bracket = (ix>0)&(ix<len(t))&v[lo]&v[hi]&((t[hi]-t[lo])<=MAX_MEASUREMENT_GAP)&(
        np.minimum(np.abs(TIME-t[lo]),np.abs(TIME-t[hi]))<=MAX_NEAREST_GAP)
    y = np.full(TIME.shape,np.nan)
    y[bracket] = np.interp(TIME[bracket], t[v], d[v])
    y[exact] = d[nearest[exact]]
    return y


def _window(y, t, a, b, min_fraction=.5):
    x = y[(t>=a)&(t<b)]
    return x[np.isfinite(x)] if len(x) and np.isfinite(x).mean()>=min_fraction else np.array([])


def _median(y,t,a,b):
    x = _window(y,t,a,b)
    return float(np.median(x)) if len(x) else np.nan


def phase_features(y, onset):
    # Each color has its own baseline and non-overlapping analysis interval.
    # Red uses 0–47s (last PIPR30 centered at 42s), blue uses 47–90s
    # (PIPR30 centered at 89s). Analysis uses NO post-blue red samples.
    a,b = ((0,47*FS) if onset == 10. else (47*FS,90*FS))
    phase=y[a:b].copy()
    t=TIME[a:b] - onset
    basewin=_window(phase,t,-5.,0.)
    if len(basewin)<MIN_BASELINE_COUNT: raise ValueError(f'Insufficient baseline at {onset}s')
    base=float(np.median(basewin))
    if not 1.<=base<=9.: raise ValueError('Invalid baseline')
    early=(t>=-2)&(t<16)
    cut=phase[early].copy()
    if np.isfinite(cut).mean()<MIN_EARLY_COVERAGE: raise ValueError(f'Insufficient CNN window at {onset}s')
    gaps=_runs(~np.isfinite(cut))
    if any((j-i)/FS>MAX_CNN_GAP_S for i,j in gaps): raise ValueError(f'Long gap in CNN window at {onset}s')
    if any(i==0 or j==len(cut) for i,j in gaps):
        # At boundaries only, up to 1 s of nearest-value padding for CNN input.
        if any((j-i)/FS>1.0 for i,j in gaps if i==0 or j==len(cut)):
            raise ValueError(f'Unobserved CNN window boundary at {onset}s')
    good=np.flatnonzero(np.isfinite(cut))
    cnn_d=np.interp(np.arange(len(cut)),good,cut[good])
    cnn_d=gaussian_filter1d(cnn_d,.9,mode='nearest')
    # Baseline-demand normalized contraction, positive = constriction.
    rel=(base-cnn_d)/base
    vel=np.clip(np.gradient(rel,DT),-3.,3.)
    # Biomarkers use only measured/locally resampled valid segments; no held or gap-filled values.
    s=_segment_smooth(phase,.9)
    peak_region=(t>=.15)&(t<=5.)
    pidx=np.flatnonzero(peak_region&np.isfinite(s))
    amp=peak_t=lat=vc=vd=rec=np.nan
    if len(pidx)>=.7*peak_region.sum():
        pk=pidx[np.argmin(s[pidx])]; amp=float(base-s[pk]); peak_t=float(t[pk])
        if amp>.04:
            threshold=max(.05,.10*amp)
            above=(t>=0)&(t<=peak_t)&np.isfinite(s)&((base-s)>=threshold)
            candidates=[i for i,j in _runs(above) if j-i>=3]
            if candidates:
                ii=candidates[0]
                # A gap just before threshold crossing makes onset ambiguous.
                if np.isfinite(s[max(0,ii-3):ii+1]).all(): lat=float(t[ii])
            deriv=np.full_like(s,np.nan)
            for i,j in _runs(np.isfinite(s)):
                if j-i>=5: deriv[i+1:j-1]=np.gradient(s[i:j],DT)[1:-1]
            neg=deriv[(t>=0)&(t<=peak_t)&np.isfinite(deriv)]
            pos=deriv[(t>=peak_t)&(t<=peak_t+6)&np.isfinite(deriv)]
            if len(neg)>=3: vc=max(0.,float(-np.percentile(neg,5)))
            if len(pos)>=3: vd=max(0.,float(np.percentile(pos,95)))
            recovered=(t>=peak_t)&np.isfinite(s)&(s>=base-.25*amp)
            cand=[i for i,j in _runs(recovered) if j-i>=5]
            if cand:
                ii=cand[0]
                # A missing interval could conceal an earlier crossing; do not invent a time.
                if np.isfinite(s[pk:ii+1]).mean()>=.95 and np.isfinite(s[max(pk,ii-1):ii+1]).all(): rec=float(t[ii]-peak_t)
        else:
            amp=max(0.,amp); peak_t=np.nan
    vals=[base,amp,100*amp/base,lat,peak_t,vc,vd,rec]
    for sec in (6,10,20,30):
        # 1-second centered median, explicitly relative to LIGHT OFFSET.
        d=_median(phase,t,PULSE_S+sec-.5,PULSE_S+sec+.5)
        vals.append(100*(base-d)/base if np.isfinite(d) else np.nan)
    ar=(t>=PULSE_S)&(t<=PULSE_S+30)
    tt=t[ar]; q=100*(base-phase[ar])/base
    ok=np.isfinite(q)
    # Trapezoids require two adjacent valid samples; normalize to 30 seconds
    # only with >=90% time coverage (an observed-time average, not a filled gap).
    adj=ok[:-1]&ok[1:]
    cov=float(np.diff(tt)[adj].sum())
    auc=float(np.sum((q[:-1][adj]+q[1:][adj])*.5*np.diff(tt)[adj])*30/cov) if cov>=27 else np.nan
    vals.append(auc)
    return np.asarray(vals,np.float32), rel.astype(np.float32), vel.astype(np.float32), float(np.isfinite(cut).mean())


def prepare_pair(left_df,right_df):
    """Return red (4,360), blue (4,360), biomarkers (39,), quality metadata.
    Missing scalar biomarkers are NaN until train-fitted median imputation.
    Raises ValueError if core waveform quality is inadequate; no forced diagnosis.
    """
    le,reye=resample_eye(left_df),resample_eye(right_df)
    means=[]; asy=[]; waves=[]; quality={}
    for phase,onset in zip(('red','blue'),ONSETS):
        lf,lr,lv,lq=phase_features(le,onset); rf,rr,rv,rq=phase_features(reye,onset)
        f=np.stack([lf,rf]); count=np.isfinite(f).sum(axis=0)
        # Both eyes required to avoid changing the feature definition by case.
        avg=np.where(count==2,np.nansum(f,axis=0)/2,np.nan)
        means.append(avg); asy.append(np.array([abs(lf[FEATURES.index(k)]-rf[FEATURES.index(k)]) for k in ASYMMETRY]))
        waves.append(np.stack([lr,rr,lv,rv]).astype(np.float32))
        quality[f'{phase}_LE_valid_fraction']=lq; quality[f'{phase}_RE_valid_fraction']=rq
    red,blue=means; ip=FEATURES.index('peak_constriction_pct'); i6=FEATURES.index('pipr6_pct'); i20=FEATURES.index('pipr20_pct')
    ratio=blue[ip]/red[ip] if np.isfinite(red[ip]) and red[ip]>2 else np.nan
    bio=np.concatenate([*means,*asy,[ratio,blue[i6]-red[i6],blue[i20]-red[i20]]]).astype(np.float32)
    quality['biomarker_valid_fraction']=float(np.isfinite(bio).mean())
    if quality['biomarker_valid_fraction']<.60: raise ValueError('Too many unavailable biomarkers')
    return waves[0],waves[1],bio,quality


def impute_biomarkers(bio, median):
    """Use the training-fitted median from model_config_v4_90s.json."""
    bio=np.asarray(bio,np.float32); med=np.asarray(median,np.float32)
    if bio.shape[-1]!=len(FEATURE_NAMES) or med.shape!=(len(FEATURE_NAMES),): raise ValueError('Biomarker schema mismatch')
    return np.where(np.isfinite(bio),bio,med).astype(np.float32)


def prepare_onnx_inputs(left_csv,right_csv,config):
    """Return batch-1 ONNX input dictionary and quality report. Normalization is inside ONNX."""
    if config.get('feature_names')!=FEATURE_NAMES: raise ValueError('Wrong model feature schema')
    if config.get('duration_s')!=90 or config.get('pulse_duration_s')!=2 or list(config.get('stimulus_onsets_s',[]))!=[10.,57.]:
        raise ValueError('Wrong model protocol: requires 90 s, red 10–12, blue 57–59')
    r,b,f,q=prepare_pair(pd.read_csv(left_csv),pd.read_csv(right_csv))
    return {'red_waveform':r[None], 'blue_waveform':b[None],
            'biomarkers':impute_biomarkers(f,config['biomarker_median'])[None]},q
