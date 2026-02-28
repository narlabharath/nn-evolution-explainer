"""
numpy_manual_gradients.py
Toy text classifier in pure NumPy with manual backprop.

Task: classify short token sequences into 2 classes.
  x: (B, T)  int  — token IDs
  E: (V, D)  float — embedding table (fixed here for simplicity)
  W: (D, C)  float — linear classifier weight
  b: (1, C)  float — bias

Forward:
  h = mean(E[x], axis=seq_len)  # (B, D)  mean-pool embeddings
  logits = h @ W + b            # (B, C)
  probs = softmax(logits)       # (B, C)
  loss = cross_entropy(probs, y)

Backward (hand-derived):
  dlogits = probs - one_hot(y)  / B
  dW = h.T @ dlogits
  db = dlogits.sum(axis=0, keepdims=True)
"""

import numpy as np

# ── Hyper-parameters ────────────────────────────────────────
B, T, V, D, C = 4, 6, 50, 8, 2   # batch, seq_len, vocab, embed_dim, classes
lr = 0.05
EPOCHS = 100

# ── Data (random for illustration) ─────────────────────────
rng = np.random.default_rng(0)
x = rng.integers(0, V, size=(B, T))        # token IDs
y = rng.integers(0, C, size=(B,))          # class labels

# ── Parameters ──────────────────────────────────────────────
E = rng.standard_normal((V, D)) * 0.1      # embedding table (kept fixed here)
W = rng.standard_normal((D, C)) * 0.1      # classifier weight
b = np.zeros((1, C))                        # classifier bias


def softmax(z):
    """Numerically stable row-wise softmax."""
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def cross_entropy(probs, targets):
    """Mean negative log-likelihood for the true class."""
    return -np.log(probs[np.arange(len(targets)), targets] + 1e-9).mean()


for epoch in range(EPOCHS):
    # ── Forward ─────────────────────────────────────────────
    h = E[x].mean(axis=1)                          # (B, D)  mean-pool
    logits = h @ W + b                             # (B, C)
    probs = softmax(logits)                        # (B, C)
    loss = cross_entropy(probs, y)

    # ── Backward (manual chain rule) ─────────────────────────
    dlogits = probs.copy()
    dlogits[np.arange(B), y] -= 1                  # d(CE)/d(logits)
    dlogits /= B                                   # mean over batch

    dW = h.T @ dlogits                             # (D, C)
    db = dlogits.sum(axis=0, keepdims=True)        # (1, C)
    # dE not computed (embeddings treated as fixed)

    # ── Update ──────────────────────────────────────────────
    W -= lr * dW
    b -= lr * db

    if epoch % 10 == 0:
        preds = logits.argmax(axis=1)
        acc = (preds == y).mean()
        print(f"Epoch {epoch:3d}  loss={loss:.4f}  acc={acc:.2f}")
