import torch
from torch.utils.data import DataLoader
from torch import nn, optim

from dataset import RiceDataset, get_transforms
from model import get_model

DATA_DIR = "data/processed"
BATCH_SIZE = 8
EPOCHS = 10
LR = 1e-3

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = RiceDataset(DATA_DIR, transform=get_transforms())
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = get_model().to(device)

    # Handle imbalance
    weights = torch.tensor([198/27, 1.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), "models/baseline_model.pth")
    print("Baseline model saved.")

if __name__ == "__main__":
    train()
