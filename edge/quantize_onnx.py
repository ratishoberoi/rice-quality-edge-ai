from onnxruntime.quantization import quantize_dynamic, QuantType

quantize_dynamic(
    model_input="models/rice_quality_edge.onnx",
    model_output="models/rice_quality_edge_int8.onnx",
    weight_type=QuantType.QInt8
)

print("INT8 QUANTIZATION DONE")
