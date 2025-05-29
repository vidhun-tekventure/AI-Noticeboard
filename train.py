import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import NoticeboardDataset
from model import NoticeboardModel

# Paths
image_dir = "data/images"
label_json = "data/labels.json"

# Load data
dataset = NoticeboardDataset(image_dir, label_json)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Setup
model = NoticeboardModel(num_labels=6)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training loop
for epoch in range(50):
    total_loss = 0
    for imgs, labels in dataloader:
        preds = model(imgs)
        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}")

# Save model
torch.save(model.state_dict(), "notice_model.pt")
