import time
import numpy as np
import onnxruntime as ort

MODEL_PATH = "edge/optimized/rice_quality_edge_fp16.onnx"

session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])

# FP16 input (VERY IMPORTANT)
dummy = np.random.rand(1, 3, 224, 224).astype(np.float16)

# Warmup
for _ in range(5):
    session.run(None, {"input": dummy})

# Benchmark
runs = 100
start = time.time()
for _ in range(runs):
    session.run(None, {"input": dummy})
end = time.time()

avg_ms = ((end - start) / runs) * 1000
print(f"🚀 Avg Inference Time: {avg_ms:.2f} ms")
