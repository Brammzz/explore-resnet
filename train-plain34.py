import os
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

from plain34 import create_plain34   # model Plain-34

# -----------------------
# Custom Dataset pakai CSV
# -----------------------
class FoodDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = self.annotations.iloc[idx, 0]   # kolom filename
        label_name = self.annotations.iloc[idx, 1] # kolom label

        # path gambar
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        # label ke angka
        label = self.label_to_index(label_name)

        if self.transform:
            image = self.transform(image)

        return image, label

    def label_to_index(self, label_name):
        # mapping manual sesuai dataset
        classes = ["gado_gado", "rendang", "bakso", "soto_ayam", "nasi_goreng"]
        return classes.index(label_name)

# -----------------------
# Hyperparameters
# -----------------------
batch_size = 32
learning_rate = 0.001
num_epochs = 20
num_classes = 5

# -----------------------
# Data Transform
# -----------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# -----------------------
# Dataset & Split (80:20)
# -----------------------
dataset = FoodDataset(csv_file="data/train.csv",
                      root_dir="data/train",
                      transform=transform)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# -----------------------
# Model, Loss, Optimizer
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = create_plain34(num_classes=num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# -----------------------
# Training Loop
# -----------------------
for epoch in range(num_epochs):
    # ---- Training ----
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_loss = running_loss / len(train_loader.dataset)
    train_acc = 100. * correct / total

    # ---- Validation ----
    model.eval()
    val_running_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()

    val_loss = val_running_loss / len(val_loader.dataset)
    val_acc = 100. * val_correct / val_total

    # log epoch
    print(f"Epoch [{epoch+1}/{num_epochs}] "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

# -----------------------
# Save model
# -----------------------
torch.save(model.state_dict(), "plain34_baseline.pth")
print("Model saved as plain34_baseline.pth")
