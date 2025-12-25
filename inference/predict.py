import argparse
import cv2
import numpy as np
import onnxruntime as ort
import os

MODEL_PATH = "edge/optimized/rice_quality_edge_fp16.onnx"
CLASSES = ["GOOD", "BAD"]

def preprocess(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Failed to load image")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # CHW
    img = np.expand_dims(img, axis=0)
    return img.astype(np.float16)

def main(image_path):
    session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    img = preprocess(image_path)
    outputs = session.run(None, {input_name: img})
    pred = int(np.argmax(outputs[0]))

    print(f"🧠 Prediction: {CLASSES[pred]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    args = parser.parse_args()
    main(args.image)
