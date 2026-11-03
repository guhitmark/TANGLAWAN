# build_trt_engine.py  (standalone, ONNX only: no PyTorch needed on the Jetson)
# ---------------------------------------------------------------
# Builds a TensorRT FP16 engine from the .onnx file the Colab notebook exported.
# Must be run ON the Jetson (an engine is tied to the GPU + TensorRT version
# that built it). Put the .onnx next to this file, edit ONNX_FILE, run the file.
# Close the browser and other apps while it builds (several minutes).
# ---------------------------------------------------------------
import os
import sys

import tensorrt as trt

# ======================= SETTINGS (edit these) =================
HERE         = os.path.dirname(os.path.abspath(__file__))   # folder this script lives in
ONNX_FILE    = "pupil_lite_400x400.onnx"     # from ritnet_lite (lite or tiny)
WORKSPACE_MB = 512
# The engine gets the same name with _fp16.engine; put that name in tracker_config.ENGINE_FILE
# ===============================================================

ONNX_PATH = os.path.join(HERE, ONNX_FILE)
ENGINE_PATH = os.path.join(HERE, os.path.splitext(ONNX_FILE)[0] + "_fp16.engine")

if not os.path.exists(ONNX_PATH):
    sys.exit("[BUILD] ONNX not found: %s" % ONNX_PATH)

logger = trt.Logger(trt.Logger.INFO)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, logger)

with open(ONNX_PATH, "rb") as f:
    if not parser.parse(f.read()):
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        sys.exit("[BUILD] ONNX parse failed")

inp, out = network.get_input(0), network.get_output(0)
print("[BUILD] input  %s %s" % (inp.name, tuple(inp.shape)))
print("[BUILD] output %s %s" % (out.name, tuple(out.shape)))
if inp.name != "input" or out.name != "logits":
    sys.exit("[BUILD] tensor names must be 'input' and 'logits' (the tracker binds those names)")

config = builder.create_builder_config()
config.max_workspace_size = WORKSPACE_MB << 20
if builder.platform_has_fast_fp16:
    config.set_flag(trt.BuilderFlag.FP16)
else:
    print("[BUILD] WARNING: no fast FP16 on this device, building FP32")

print("[BUILD] Building engine, workspace %d MB..." % WORKSPACE_MB)
blob = builder.build_serialized_network(network, config)
if blob is None:
    sys.exit("[BUILD] Engine build failed")

with open(ENGINE_PATH, "wb") as f:
    f.write(blob)
print("[BUILD] Engine saved -> %s" % ENGINE_PATH)
print("[BUILD] Next: set ENGINE_FILE = \"%s\" in tracker_config.py" % os.path.basename(ENGINE_PATH))