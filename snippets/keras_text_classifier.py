"""
keras_text_classifier.py
Toy text classifier using TF-Keras.

Task: classify padded token sequences into 2 classes.
  x: (B, T)  int  — padded token IDs
  Model: Embedding → GlobalAveragePooling1D → Dense → Dense (logits)

Run with:
    pip install tensorflow
    python snippets/keras_text_classifier.py
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# ── Config ──────────────────────────────────────────────────
V, D, T, C = 20_000, 128, 64, 2         # vocab, embed_dim, max_len, classes
BATCH_SIZE = 32
EPOCHS = 3

# ── Toy dataset (random token IDs + labels) ─────────────────
np.random.seed(0)
N = 512
x_data = np.random.randint(0, V, size=(N, T))
y_data = np.random.randint(0, C, size=(N,))

# Build a tf.data.Dataset
ds_full = tf.data.Dataset.from_tensor_slices((x_data, y_data))
ds_full = ds_full.shuffle(N, seed=42)
train_ds = ds_full.take(400).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_ds   = ds_full.skip(400).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# ── Model definition ────────────────────────────────────────
model = tf.keras.Sequential([
    layers.Input(shape=(T,), dtype="int32"),
    layers.Embedding(V, D),               # (B, T, D)
    layers.GlobalAveragePooling1D(),      # (B, D)  ← mean-pool over T
    layers.Dense(64, activation="relu"),  # (B, 64)
    layers.Dense(C),                      # (B, C)  logits
], name="text_classifier")

model.summary()

# ── Compile ─────────────────────────────────────────────────
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(
    optimizer="adam",
    loss=loss_fn,
    metrics=["accuracy"],
)

# ── Train (fit() runs: forward → loss → backward → update) ──
model.fit(train_ds, epochs=EPOCHS, validation_data=val_ds)

# ── Evaluate ────────────────────────────────────────────────
loss, acc = model.evaluate(val_ds, verbose=0)
print(f"\nVal loss: {loss:.4f}  Val acc: {acc:.4f}")

# ── Save (TF SavedModel format) ──────────────────────────────
model.save("assets/checkpoints/keras_text_classifier.keras")
print("Model saved.")
