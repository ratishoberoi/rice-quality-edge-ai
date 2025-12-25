import os
import cv2
import numpy as np
import onnxruntime as ort

MODEL_PATH = "edge/optimized/rice_quality_edge_fp16.onnx"
DATA_DIR = "data/processed"
IMG_SIZE = 224

session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])

correct = 0
total = 0

for label, cls in enumerate(["bad", "good"]):
    folder = os.path.join(DATA_DIR, cls)
    for img_name in os.listdir(folder):
        img_path = os.path.join(folder, img_name)

        img = cv2.imread(img_path)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img.astype(np.float16) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)

        outputs = session.run(None, {"input": img})
        pred = np.argmax(outputs[0])

        if pred == label:
            correct += 1
        total += 1

print(f"✅ Edge Model Accuracy: {100 * correct / total:.2f}% ({correct}/{total})")
