# Era 2 — Graph-first / Symbolic (Theano / TF1-style pseudocode)
# ---------------------------------------------------------------
# NOTE: This is *conceptual* pseudocode illustrating the two-phase
# (define-graph → run-graph) programming model.
# It is NOT directly runnable; actual TF1 syntax differs slightly.

# ─── Phase 1: Build the graph (no data flows yet) ───────────────
x     = placeholder(shape=(None, T), dtype=int32)    # symbolic input — token IDs
E_var = variable(shape=(V, D))                        # embedding table (trainable)
W     = variable(shape=(D, C))                        # output projection
b     = variable(shape=(C,))

h      = mean(embedding_lookup(E_var, x), axis=1)    # (B, D) — symbolic op
logits = matmul(h, W) + b                            # (B, C) — symbolic op
loss   = cross_entropy_from_logits(logits, y)        # scalar — symbolic op

# Gradient ops are built automatically on the graph (autodiff)
train_op = optimizer("adam").minimize(loss)          # creates grad + update ops

# ─── Phase 2/3: Compile-then-run ────────────────────────────────
with Session() as sess:
    sess.run(init_all_variables())
    for batch_x, batch_y in data_loader():
        sess.run(train_op, feed_dict={x: batch_x, y: batch_y})

# Key observations:
#   • placeholder() represents symbolic inputs — no real data yet
#   • minimize(loss) builds gradient + update ops on the graph
#   • You can't print or inspect tensors before session.run()
#   • The runtime can optimize how and where ops execute
