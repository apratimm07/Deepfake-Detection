# =====================================================
#   DeepFake Detection - EfficientNet-B0 Fine-Tuning
#   Full Training + Evaluation + Loss Graph
#   Author: Apratim Mishra, IIT Jammu
# =====================================================

import torch, torch.nn as nn, torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
import numpy as np
import os
from PIL import Image

# -----------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------
DATA_DIR = r"C:\Users\Apratim\cs\Deepfake\Celeb-DF-v2"
REAL_DIR = os.path.join(DATA_DIR, "faces_real")
FAKE_DIR = os.path.join(DATA_DIR, "faces_synthetic")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 10
BATCH_SIZE = 16
LR = 1e-4

print(f"⚙️ Using device: {DEVICE}")

# -----------------------------------------------------
# CUSTOM DATASET
# -----------------------------------------------------
class DeepFakeDataset(Dataset):
    def __init__(self, real_dir, fake_dir, transform=None):
        self.real_imgs = [os.path.join(real_dir, f) for f in os.listdir(real_dir)
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        self.fake_imgs = [os.path.join(fake_dir, f) for f in os.listdir(fake_dir)
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        self.transform = transform
        self.data = [(p, 0) for p in self.real_imgs] + [(p, 1) for p in self.fake_imgs]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path, label = self.data[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


# -----------------------------------------------------
# TRANSFORMS
# -----------------------------------------------------
train_tfm = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.1,0.1,0.1,0.05),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])
test_tfm = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# -----------------------------------------------------
# LOAD DATASET
# -----------------------------------------------------
full_dataset = DeepFakeDataset(REAL_DIR, FAKE_DIR, transform=train_tfm)
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_set, test_set = random_split(full_dataset, [train_size, test_size])

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

print(f"🧠 Total images: {len(full_dataset)} | Train: {len(train_set)}, Test: {len(test_set)}")

# -----------------------------------------------------
# MODEL SETUP
# -----------------------------------------------------
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, 1)
model = model.to(DEVICE)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# -----------------------------------------------------
# TRAINING LOOP
# -----------------------------------------------------
print("\n🚀 Starting fine-tuning...\n")
train_losses = []

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        imgs, labels = imgs.to(DEVICE), labels.float().unsqueeze(1).to(DEVICE)

        # Forward
        out = model(imgs)
        loss = criterion(out, labels)

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)
    print(f"📉 Epoch {epoch+1}/{EPOCHS} | Avg Loss: {avg_loss:.4f}")

# -----------------------------------------------------
# SAVE MODEL
# -----------------------------------------------------
SAVE_PATH = os.path.join(DATA_DIR, "deepfake_effb0_finetuned.pth")
torch.save(model.state_dict(), SAVE_PATH)
print(f"\n✅ Model saved at: {SAVE_PATH}")

# -----------------------------------------------------
# LOSS GRAPH
# -----------------------------------------------------
plt.figure(figsize=(8,5))
plt.plot(range(1, EPOCHS+1), train_losses, marker='o', linestyle='-')
plt.title("Training Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Binary Cross-Entropy Loss")
plt.grid(True)
plt.show()

# -----------------------------------------------------
# EVALUATION
# -----------------------------------------------------
model.eval()
y_true, y_pred, y_prob = [], [], []

with torch.no_grad():
    for imgs, labels in test_loader:
        imgs = imgs.to(DEVICE)
        logits = model(imgs)
        probs = torch.sigmoid(logits).cpu().numpy().flatten()
        preds = (probs > 0.5).astype(int)
        y_true.extend(labels.numpy())
        y_pred.extend(preds)
        y_prob.extend(probs)

# -----------------------------------------------------
# METRICS
# -----------------------------------------------------
acc = accuracy_score(y_true, y_pred)
auc = roc_auc_score(y_true, y_prob)
cm = confusion_matrix(y_true, y_pred)

print(f"\n✅ Accuracy: {acc*100:.2f}% | AUROC: {auc:.3f}")
print("Confusion Matrix:\n", cm)

# -----------------------------------------------------
# PLOT CONFUSION MATRIX
# -----------------------------------------------------
plt.figure(figsize=(5,4))
plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha="center", va="center", color="black", fontsize=12)
plt.colorbar()
plt.show()
