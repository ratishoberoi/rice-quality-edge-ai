import onnx
from onnxconverter_common import float16

INPUT_MODEL = "models/rice_quality_baseline_single.onnx"
OUTPUT_MODEL = "edge/optimized/rice_quality_edge_fp16.onnx"

print("Loading ONNX model...")
model = onnx.load(INPUT_MODEL)

print("Converting to FP16...")
model_fp16 = float16.convert_float_to_float16(model)

onnx.save(model_fp16, OUTPUT_MODEL)

print("FP16 Edge model saved →", OUTPUT_MODEL)
