# Neural Networks — NumPy to PyTorch

Interactive single-page HTML explainer for the IISc × TalentSprint GenAI programme.

## How to open

Just open `index.html` directly in any modern browser:

```
# macOS / Linux
open index.html

# Windows
start index.html
```

No build step, no local server required.

## Regenerating the plots

Three illustrative PNG figures are embedded in the page. They are produced by a single matplotlib script:

```bash
pip install matplotlib numpy        # one-time dependency
cd nn-evolution-doc
python scripts/make_plots.py
```

Outputs written to `assets/plots/`:

| File | Used in |
|------|---------|
| `loss_curve.png` | Section E — Training Loop |
| `tradeoff_scatter.png` | Section J — Why PyTorch |
| `cognitive_load.png` | Section J — Why PyTorch |

## Diagrams (Mermaid)

Diagrams are rendered client-side via **Mermaid.js** loaded from jsDelivr CDN:

```
https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js
```

- **Online:** diagrams render automatically.  
- **Offline:** the rest of the page works fine; diagram containers will be empty.

## Project structure

```
nn-evolution-doc/
├── index.html                  ← Main deliverable (open this)
├── css/
│   └── styles.css              ← Fluent/Microsoft design system
├── js/
│   └── main.js                 ← Mermaid init, copy buttons, nav highlighting
├── scripts/
│   └── make_plots.py           ← matplotlib figure generator
├── snippets/
│   ├── numpy_manual_gradients.py      ← Era 1: raw NumPy forward pass
│   ├── graph_first_pseudocode.py      ← Era 2: define-graph-then-run (conceptual)
│   ├── eager_autograd_demo.py         ← Era 4: requires_grad + loss.backward()
│   ├── keras_text_classifier.py       ← Era 3/5: Keras compile/fit style
│   ├── pytorch_text_classifier.py     ← Era 5: full explicit PyTorch loop
│   └── torch_compile_upgrade.py       ← Era 6: one-line torch.compile wrapper
└── assets/
    ├── logos/                  ← SVG logos: numpy, tensorflow, keras, pytorch
    └── plots/                  ← Generated PNGs (git-ignored if large)
```

## Sections covered

| # | Section | Topic |
|---|---------|-------|
| A | Hero | Overview + key stats |
| B | Agenda | Session roadmap |
| C | NN Basics | Forward pass, activations, vocab |
| D | 7 Jobs | What every framework must handle |
| E | Training Loop | Forward → loss → backward → update |
| F | Why Python | Vectorisation, ecosystem, device portability |
| G | Timeline | 2011 → 2024 evolution timeline |
| H | Era Cards | 6 eras with code + diagrams (NumPy → torch.compile) |
| I | Comparison | Keras ↔ PyTorch concept mapping, decision matrix, scenarios |
| J | Why PyTorch | Four forces: debugging, loops, ecosystem, performance |
| K | Course note | Keras today → PyTorch from next module |
