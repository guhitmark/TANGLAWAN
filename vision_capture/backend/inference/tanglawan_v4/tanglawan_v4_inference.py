"""TANGLAWAN V4 90s research inference adapter.

Requires model_config_v4_90s.json and a model from the SAME trained run.
No clinical diagnosis is established by this module. Does not use staged
hold/interpolated values as measured pupils.

Example:
  python tanglawan_v4_inference.py --model-dir ./models_v4_90s_hybrid_V2_sanitycheck \
    --left left.csv --right right.csv
"""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

from .tanglawan_plr_features_90s import (
    FEATURE_NAMES, prepare_onnx_inputs, prepare_pair, impute_biomarkers,
)

MODEL_BASENAME = 'tanglawan_1dcnn_v4_90s'
CONFIG_NAME = 'model_config_v4_90s.json'


class V4Inference:
    def __init__(self, model_dir, backend='auto'):
        self.root = Path(model_dir).expanduser().resolve()
        path = self.root / CONFIG_NAME
        if not path.is_file():
            raise FileNotFoundError('Missing '+str(path)+'; copy the JSON from the same training run')
        self.config = json.loads(path.read_text(encoding='utf-8'))
        c = self.config
        if (c.get('duration_s') != 90 or c.get('pulse_duration_s') != 2 or
                list(c.get('stimulus_onsets_s', [])) != [10., 57.]):
            raise ValueError('Model config does not match the 90-second red 10–12 / blue 57–59 protocol')
        if c.get('feature_names') != FEATURE_NAMES:
            raise ValueError('Feature schema mismatch: use the preprocessing module from the same V4 notebook')
        if c.get('input_names') != ['red_waveform', 'blue_waveform', 'biomarkers']:
            raise ValueError('Unsupported model input names')
        if c.get('input_shapes') != {'red_waveform':[1,4,360],'blue_waveform':[1,4,360],'biomarkers':[1,39]}:
            raise ValueError('Unsupported model input shapes')
        if c.get('training_data_mode') != 'synthetic':
            # Other training modes may be legitimate; do not silently assume its provenance.
            print('NOTICE: training_data_mode =', repr(c.get('training_data_mode')))
        model_onnx = self.root / (MODEL_BASENAME+'.onnx')
        model_ts = self.root / (MODEL_BASENAME+'.ts')
        if backend == 'auto':
            # ONNX Runtime may be absent on Jetson Nano. Prefer ONNX when it
            # actually loads; otherwise try TorchScript if available.
            if model_onnx.is_file():
                try:
                    import onnxruntime as ort
                    self.model = ort.InferenceSession(str(model_onnx), providers=['CPUExecutionProvider'])
                    actual = {i.name: list(i.shape) for i in self.model.get_inputs()}
                    if actual != c['input_shapes']:
                        raise ValueError('ONNX input shapes differ from config: '+repr(actual))
                    self.backend = 'onnx'
                    return
                except Exception as exc:
                    if not model_ts.is_file():
                        raise RuntimeError('ONNX backend could not start and TorchScript fallback is missing') from exc
                    print('[V4] ONNX Runtime unavailable/incompatible; using TorchScript:', exc)
            backend = 'torchscript' 
        self.backend = backend
        if backend == 'onnx':
            if not model_onnx.is_file(): raise FileNotFoundError(str(model_onnx))
            try:
                import onnxruntime as ort
            except ImportError as exc:
                raise ImportError('ONNX Runtime is unavailable; install a version compatible with this Jetson, or use --backend torchscript') from exc
            self.model = ort.InferenceSession(str(model_onnx), providers=['CPUExecutionProvider'])
            actual = {i.name: list(i.shape) for i in self.model.get_inputs()}
            expected = c['input_shapes']
            if actual != expected:
                raise ValueError('ONNX input shapes differ from config: '+repr(actual))
        elif backend == 'torchscript':
            if not model_ts.is_file(): raise FileNotFoundError(str(model_ts))
            import torch
            self.torch = torch
            self.model = torch.jit.load(str(model_ts), map_location='cpu').eval()
        else:
            raise ValueError('backend must be auto, onnx, or torchscript')

    def _infer(self, feeds):
        if self.backend == 'onnx':
            out = self.model.run(None, feeds)[0]
        else:
            torch = self.torch
            with torch.inference_mode():
                out = self.model(*(torch.from_numpy(feeds[name]) for name in self.config['input_names'])).numpy()
        logits = np.asarray(out, dtype=np.float64)
        if logits.shape != (1,3) or not np.isfinite(logits).all():
            raise RuntimeError('Unexpected model output '+str(logits.shape))
        logits = logits[0]
        s = np.exp(logits - np.max(logits))
        return s / s.sum()

    def _result(self, feeds, quality):
        p = self._infer(feeds)
        class_to_index = self.config['class_to_index']
        inverse = {int(index):label for label,index in class_to_index.items()}
        idx = int(np.argmax(p)); label = inverse[idx]
        return {
            'ok': True,
            'research_only': True,
            'model_version': self.config.get('model_version'),
            'training_data_mode': self.config.get('training_data_mode'),
            'label': label,
            'class_name': self.config.get('class_names',{}).get(label, label),
            'model_score': float(p[idx]),
            'scores_by_label': {inverse[i]:float(p[i]) for i in range(3)},
            'quality': quality,
            'warning': 'Synthetic sanity-check model: not clinically validated. Scores are not calibrated disease probabilities.',
        }

    def predict_csv(self, left_csv, right_csv):
        feeds, quality = prepare_onnx_inputs(str(left_csv), str(right_csv), self.config)
        return self._result(feeds, quality)

    def predict_dataframes(self, left, right):
        """Use actual measured rows with original elapsed timestamps, valid flags,
        and optionally confidence. Never use display-only held values as measurements.
        """
        red, blue, bio, quality = prepare_pair(left, right)
        feeds = {
            'red_waveform':red[None].astype(np.float32),
            'blue_waveform':blue[None].astype(np.float32),
            'biomarkers':impute_biomarkers(bio, self.config['biomarker_median'])[None],
        }
        return self._result(feeds, quality)

    def predict_session(self, state):
        """Run only after a completed 90s recording. Both eyes must have data.
        State records are expected as {'t':sec,'y':diameter_mm,'valid':bool}.
        Missing confidence is not fabricated; downstream quality checks still apply.
        """
        if getattr(state, 'is_recording', False):
            raise ValueError('Recording still running: inference is allowed only after completion')
        if (not getattr(state, 'has_staged_data', False) and float(getattr(state, 'frozen_elapsed', 0)) < 89) or max(float(state.elapsed()), float(getattr(state, 'frozen_elapsed', 0))) < 89:
            raise ValueError('Require a completed 90-second recording with preserved session data')
        if getattr(state,'tracker_mode',False):
            raise ValueError('Tracker mode is not the 90-second clinical protocol')
        with state.lock:
            left = list(state.session_data.get('L', []))
            right = list(state.session_data.get('R', []))
        if not left or not right:
            raise ValueError('Both recorded eyes are required by this V4 model')
        def convert(rows):
            return pd.DataFrame({
                'timestamp_s':[r.get('t') for r in rows],
                'diameter_mm':[r.get('y') for r in rows],
                'valid':[r.get('valid',False) for r in rows],
            })
        return self.predict_dataframes(convert(left), convert(right))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir', required=True, help='Folder containing config plus ONNX or TorchScript model')
    p.add_argument('--left', required=True, help='Left-eye 90s CSV')
    p.add_argument('--right', required=True, help='Right-eye 90s CSV')
    p.add_argument('--backend', choices=['auto','onnx','torchscript'],default='auto')
    args = p.parse_args()
    try:
        result = V4Inference(args.model_dir, args.backend).predict_csv(args.left,args.right)
        print(json.dumps(result, indent=2))
    except (ValueError,FileNotFoundError,RuntimeError,ImportError) as exc:
        print(json.dumps({'ok':False,'error':str(exc),'repeat_or_inspect_recording':True},indent=2))
        raise SystemExit(1)

if __name__ == '__main__':
    main()
