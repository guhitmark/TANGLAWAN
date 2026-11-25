"""TANGLAWAN 90 s V4 research inference and report integration.

Replacement for the old randomized inference.py. Existing routes can keep:
    from ..inference import run_inference

Do not reload the ONNX/TS model for every request. Never run from /status.
"""
import os
import threading
from pathlib import Path
from .tanglawan_v4 import V4Inference

_MODEL = None
_MODEL_LOCK = threading.Lock()


def get_model():
    global _MODEL
    if _MODEL is None:
        with _MODEL_LOCK:
            if _MODEL is None:
                model_dir = Path(__file__).resolve().parent / 'tanglawan_v4'
                backend = os.environ.get('TANGLAWAN_V4_BACKEND', 'auto').lower()
                _MODEL = V4Inference(model_dir, backend=backend)
                print('[V4] Loaded research inference backend:', _MODEL.backend)
    return _MODEL


def run_inference(state, patient_details):
    """Run V4 on a completed, saved/staged binocular session and build HTML report.

    Preserves the old signature and familiar return keys: prediction, confidence,
    report_path. The confidence key is an uncalibrated MODEL SCORE (percentage),
    not calibrated disease probability or clinical confidence.
    Inference failures intentionally raise: never fabricate a class.
    """
    if getattr(state, 'is_recording', False):
        raise ValueError('Wait for the 90-second recording to complete before generating a report')
    if getattr(state, 'tracker_mode', False):
        raise ValueError('The 60-second tracker mode is incompatible with the V4 model')
    if not getattr(state, 'left_enabled', True) or not getattr(state, 'right_enabled', True):
        raise ValueError('V4 requires both left and right eye recordings')
    result = get_model().predict_session(state)
    from ..report_generator import generate_diagnostic_report
    model_score_pct = round(100.0 * result['model_score'], 2)
    prediction = '%s (%s)' % (result['label'], result['class_name'])
    path = generate_diagnostic_report(
        state, patient_details or {}, prediction, model_score_pct,
        model_result=result,
    )
    result = dict(result)
    result.update({
        'prediction': prediction,
        'confidence': model_score_pct,  # compatibility only; NOT calibrated confidence
        'report_path': path,
    })
    return result
