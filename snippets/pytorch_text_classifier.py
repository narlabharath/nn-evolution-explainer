"""
pytorch_text_classifier.py
Toy text classifier using PyTorch — explicit training loop.

Task: classify padded token sequences into 2 classes.
  x: (B, T)  int  — padded token IDs
  Model: nn.Embedding → mean-pool → nn.Linear → logits

Run with:
    pip install torch
    python snippets/pytorch_text_classifier.py
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ── Config ──────────────────────────────────────────────────
V, D, T, C = 20_000, 128, 64, 2
BATCH_SIZE = 32
EPOCHS = 3
LR = 1e-3

# ── Device ──────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ── Toy dataset ─────────────────────────────────────────────
torch.manual_seed(0)
N = 512
x_data = torch.randint(0, V, (N, T))      # token IDs
y_data = torch.randint(0, C, (N,))        # labels

dataset = TensorDataset(x_data, y_data)
train_ds, val_ds = torch.utils.data.random_split(dataset, [400, 112])
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)


# ── Model ────────────────────────────────────────────────────
class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)  # (V, D)
        self.fc1       = nn.Linear(embed_dim, 64)
        self.relu      = nn.ReLU()
        self.fc2       = nn.Linear(64, num_classes)           # logits

    def forward(self, x):                  # x: (B, T)
        h = self.embedding(x)              # (B, T, D)
        h = h.mean(dim=1)                  # (B, D)  mean-pool
        h = self.relu(self.fc1(h))         # (B, 64)
        return self.fc2(h)                 # (B, C)  logits


model   = TextClassifier(V, D, C).to(device)
loss_fn = nn.CrossEntropyLoss()           # expects logits
opt     = torch.optim.Adam(model.parameters(), lr=LR)


# ── Training loop (explicit) ─────────────────────────────────
for epoch in range(EPOCHS):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        # 1) zero grads
        opt.zero_grad()
        # 2) forward
        logits = model(x_batch)            # (B, C)
        # 3) loss
        loss = loss_fn(logits, y_batch)
        # 4) backward + update
        loss.backward()
        opt.step()

        total_loss += loss.item() * len(y_batch)
        correct    += (logits.argmax(1) == y_batch).sum().item()
        total      += len(y_batch)

    avg_loss = total_loss / total
    acc      = correct / total
    print(f"Epoch {epoch+1}/{EPOCHS}  train_loss={avg_loss:.4f}  train_acc={acc:.4f}")

# ── Evaluation loop ──────────────────────────────────────────
model.eval()
val_correct, val_total = 0, 0
with torch.no_grad():
    for x_batch, y_batch in val_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        logits = model(x_batch)
        val_correct += (logits.argmax(1) == y_batch).sum().item()
        val_total   += len(y_batch)

print(f"\nVal acc: {val_correct/val_total:.4f}")

# ── Save (state_dict) ────────────────────────────────────────
torch.save(model.state_dict(), "assets/checkpoints/pytorch_text_classifier.pt")
print("State dict saved.")
