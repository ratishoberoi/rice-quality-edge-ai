import torch
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from training.model import get_model

DATA_DIR = "data/processed"
MODEL_PATH = "models/baseline_model.pth"
BATCH_SIZE = 16

device = "cpu"

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
])

dataset = ImageFolder(DATA_DIR, transform=transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

model = get_model()
state = torch.load(MODEL_PATH, map_location="cpu")

# 🔥 KEY FIX
model.load_state_dict(state, strict=False)

model.eval()
model.to(device)

correct = 0
total = 0

with torch.no_grad():
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        preds = out.argmax(1)
        correct += (preds == y).sum().item()
        total += y.size(0)

acc = 100 * correct / total
print(f"\n✅ Baseline Accuracy: {acc:.2f}% ({correct}/{total})")
