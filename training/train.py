import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import RiceQualityNet

# CONFIG
DATA_DIR = "data/processed"
BATCH_SIZE = 16
EPOCHS = 5
LR = 1e-4
IMG_SIZE = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# TRANSFORMS
train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

# DATASET
dataset = datasets.ImageFolder(DATA_DIR, transform=train_tf)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# MODEL
model = RiceQualityNet(num_classes=2).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# TRAIN LOOP
for epoch in range(EPOCHS):
    model.train()
    correct, total, loss_sum = 0, 0, 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    acc = 100 * correct / total
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {loss_sum:.3f} | Acc: {acc:.2f}%")

# SAVE MODEL
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/baseline_model.pth")
print("Baseline model saved to models/baseline_model.pth")
