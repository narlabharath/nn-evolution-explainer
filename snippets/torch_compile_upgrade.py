# Era 6 — Compilation Era (torch.compile)
# -----------------------------------------
# Shows how to upgrade an existing eager PyTorch model to a compiled version
# with a single line — no changes to model code or training loop required.

import torch
import torch.nn as nn

# ─── Same model definition as Era 5 (unchanged) ─────────────────
class TextClassifier(nn.Module):
    def __init__(self, V=20_000, D=128, C=2):
        super().__init__()
        self.emb  = nn.Embedding(V, D)
        self.fc1  = nn.Linear(D, 64)
        self.relu = nn.ReLU()
        self.fc2  = nn.Linear(64, C)

    def forward(self, x):                   # x: (B, T)
        h = self.emb(x).mean(dim=1)         # (B, D)
        return self.fc2(self.relu(self.fc1(h)))

device = "cuda" if torch.cuda.is_available() else "cpu"

# ─── Before: eager model ─────────────────────────────────────────
model = TextClassifier().to(device)

# ─── After: one-line compilation (PyTorch >= 2.0) ────────────────
model = torch.compile(model)

# ─── Training loop is IDENTICAL — no changes needed ─────────────
loss_fn = nn.CrossEntropyLoss()
opt     = torch.optim.Adam(model.parameters(), lr=1e-3)

# for x_b, y_b in loader:
#     opt.zero_grad()
#     logits = model(x_b)     # runs compiled (optimized) kernel
#     loss   = loss_fn(logits, y_b)
#     loss.backward()
#     opt.step()

# Key observations:
#   • torch.compile() is a wrapper — the model definition does not change
#   • First batch is slower (tracing/compilation); subsequent batches are faster
#   • If compilation fails, fall back to eager: remove the torch.compile() line
#   • Treat compilation as an optional performance accelerator, not a requirement
