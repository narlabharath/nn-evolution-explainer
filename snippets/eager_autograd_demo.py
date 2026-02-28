# Era 4 — Eager / Define-by-run
# -------------------------------
# Tiny "feel" example: eager forward with automatic gradient computation.
# No nn.Module required — raw tensors with requires_grad=True.

import torch

# A tiny linear model: logits = x @ W + b
x = torch.randn(4, 3)                        # (B=4, d_in=3) — input batch
W = torch.randn(3, 2, requires_grad=True)    # (d_in, C) — learnable
b = torch.zeros(2,   requires_grad=True)     # (C,)       — learnable

# Forward pass (eager — executes immediately, graph recorded)
logits = x @ W + b                           # (4, 2)
loss   = (logits ** 2).mean()                # any scalar-valued loss

# Backward pass — autograd walks the recorded graph
loss.backward()                              # W.grad and b.grad now populated

# Manual update step (no optimizer — for illustration)
with torch.no_grad():                        # don't record the update itself
    W -= 0.1 * W.grad
    b -= 0.1 * b.grad
    W.grad.zero_()
    b.grad.zero_()

print("loss:", float(loss))

# Key observations:
#   • requires_grad=True tells autograd to track operations on that tensor
#   • The forward pass is normal Python — ops execute on real values immediately
#   • loss.backward() computes gradients for every tracked upstream tensor
#   • torch.no_grad() prevents the update step from being added to the graph
#   • Compare: this is conceptually what nn.Module + optimizer.step() automates
