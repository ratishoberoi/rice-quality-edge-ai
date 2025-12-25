import torch
from training.model import get_model

MODEL_PATH = "models/baseline_model.pth"
ONNX_PATH = "models/rice_quality_baseline.onnx"

def convert():
    model = get_model()
    state = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    dummy = torch.randn(1, 3, 224, 224)

    torch.onnx.export(
        model,
        dummy,
        ONNX_PATH,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=None   # VERY IMPORTANT
    )

    print("✅ ONNX export done:", ONNX_PATH)

if __name__ == "__main__":
    convert()
