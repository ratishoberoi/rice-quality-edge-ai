from onnxruntime.quantization import quantize_dynamic, QuantType

INPUT_MODEL = "models/rice_quality_baseline_single.onnx"
OUTPUT_MODEL = "models/rice_quality_edge_int8.onnx"

quantize_dynamic(
    model_input=INPUT_MODEL,
    model_output=OUTPUT_MODEL,
    weight_type=QuantType.QInt8
)

print("INT8 Edge model saved →", OUTPUT_MODEL)
