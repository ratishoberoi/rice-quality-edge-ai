import onnx
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType
import numpy as np
import cv2
import os

MODEL_PATH = "models/rice_quality_baseline_single.onnx"
OUT_PATH = "models/rice_quality_edge_int8.onnx"
CALIB_DIR = "data/processed"

class RiceCalibrationReader(CalibrationDataReader):
    def __init__(self):
        self.data = []
        for cls in ["good", "bad"]:
            folder = os.path.join(CALIB_DIR, cls)
            for img in os.listdir(folder)[:10]:
                path = os.path.join(folder, img)
                image = cv2.imread(path)
                image = cv2.resize(image, (224, 224))
                image = image.transpose(2, 0, 1)
                image = image[np.newaxis, :].astype(np.float32) / 255.0
                self.data.append({"input": image})
        self.iterator = iter(self.data)

    def get_next(self):
        return next(self.iterator, None)

quantize_static(
    model_input=MODEL_PATH,
    model_output=OUT_PATH,
    calibration_data_reader=RiceCalibrationReader(),
    weight_type=QuantType.QInt8
)

print("✅ INT8 Edge model saved:", OUT_PATH)
