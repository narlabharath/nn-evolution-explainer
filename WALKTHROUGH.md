# Session Walkthrough Guide
### Neural Networks: NumPy → PyTorch
**IISc × TalentSprint · GenAI C2**

---

## Before you begin

- Open the live page: **https://narlabharath.github.io/nn-evolution-explainer/**
- Check the nav bar loads and all section links scroll correctly
- Confirm Mermaid diagrams render (needs internet)
- Estimated session time: **90–120 minutes** (adjust per cohort pace)

---

## Section A — Hero (2 min)

**Purpose:** Set expectations. This is a context-building session, not a hands-on lab.

**Say:**
> "Before we write a single line of PyTorch, let's understand *why* things are the way they are. By the end you'll know what every framework must do — and why PyTorch is where the ecosystem landed."

**Point out:** The three chips — NumPy → Keras → PyTorch. That's the arc of today.

---

## Section B — Agenda (1 min)

**Purpose:** Orient the cohort. Let them see the full map before diving in.

**Say:**
> "We have 11 stops. Think of the first half as building vocabulary, the second half as building judgment."

**Quick scan** — don't read every item. Just confirm they can see the full list.

---

## Section C — NN Basics (10 min)

**Purpose:** Establish the irreducible vocabulary everyone needs.

### Key talking points

1. **Show the Mermaid diagram** — "Every neural network, no matter how complex, is just this: input → multiply → activate → repeat."

2. **Walk through the NumPy code** — highlight three lines:
   ```
   h      = relu(x @ W1 + b1)   # matrix multiply + activation
   logits = h @ W2 + b2          # no activation on output
   print(logits.shape)           # always check shapes
   ```
   Ask: *"Why no activation on the final layer?"* (logits → loss handles it)

3. **Activation table** — spend 60 seconds:
   - Hidden layers → ReLU (default, fast, works)
   - Final layer → nothing (raw logits; CrossEntropyLoss does softmax internally)

4. **Common confusion callout** — point it out explicitly. Logits ≠ probabilities. This trips most learners in the first lab.

**Check-in question:** *"What shape does the output of the forward pass have? What does each dimension mean?"*

---

## Section D — The 7 Jobs (12 min)

**Purpose:** Show that every framework is just solving the same 7 problems.

### Key talking points

1. **Read the 7 job cards out loud** (fast pass). Frame it:
   > "Whether you write Keras or PyTorch, these 7 things must happen. The difference is *who writes the code* for each."

2. **"Who does what" diagrams** — Keras hides jobs 3–5 inside `fit()`. PyTorch makes you write them.

3. **The irreducible 4-step core** — stop here and make it stick:
   ```
   zero_grad → forward → loss → backward → step
   ```
   Ask: *"If you forget zero_grad, what happens?"* (gradients accumulate = wrong updates)

4. **Eval loop comparison** — highlight the two easy PyTorch mistakes:
   - Forget `model.eval()` → dropout/BN still active
   - Forget `torch.no_grad()` → massive memory waste

**Pause point:** Ask if the 7 jobs make sense before moving on. This framing pays off when they see PyTorch code for the first time.

---

## Section E — Training Loop (12 min)

**Purpose:** Make the 4-step loop automatic and intuitive.

### Key talking points

1. **The loss curve plot** — explain what a healthy vs broken curve looks like:
   - Descending smoothly → working
   - Flat from start → LR too low or bug in backward
   - NaN/exploding → LR too high or missing zero_grad

2. **Walk through the annotated PyTorch loop** step by step — read each comment aloud.

3. **The gradient flow diagram** — trace it visually: output → loss → backward → each weight.

**Demo idea:** Ask learners to mentally "break" the loop — what happens if you remove each of the 4 steps?

---

## Section F — Why Python? (8 min)

**Purpose:** Justify the ecosystem choice; build appreciation for NumPy.

### Key talking points

1. **Vectorisation demo** — show the timing difference between a Python loop and `np.dot`. The point: GPUs are just this idea taken to the extreme.

2. **The three reasons** — ecosystem, device portability, readable code. Don't dwell — these are intuitive.

3. **Library ecosystem diagram** — point out the layering: NumPy at the bottom, everything else builds on it.

**Quick ask:** *"What's the difference between running on CPU vs GPU from a code perspective?"* (Almost nothing — just `.to(device)`)

---

## Section G — Timeline (10 min)

**Purpose:** Show evolution *felt* inevitable in hindsight — but each era solved a real pain.

### Key talking points

1. **Walk the timeline left to right** — don't read every event, just hit the inflection points:
   - 2012: AlexNet proves deep nets + GPUs work
   - 2015: TF1 ships → autodiff for everyone
   - 2017: PyTorch ships → ergonomics revolution
   - 2022: torch.compile → eager + speed

2. **Breakthrough table** — use this to link each era to what bottleneck it removed *and* what new one it introduced. Key insight: there's no perfect era; each trade-off led to the next.

3. **Mental model diagrams** — "The shape of computation didn't change. Only who manages it did."

**Reflection question:** *"Which era would have felt most frustrating to work in? Why?"*

---

## Section H — Era Cards (20 min)

**Purpose:** Deep-dive into each era. This is the core of the session.

**Pacing:** Spend ~3 min per era. Don't rush Era 5 (PyTorch) — that's where they'll live.

---

### Era 1 — NumPy (3 min)
- Show the manual gradient code. Ask: *"What's missing vs modern PyTorch?"*
- Answer: no autodiff. You derive and code gradients by hand for every new architecture.
- Punch line: "This is why autodiff matters — not just convenience, but correctness."

---

### Era 2 — Graph-first (3 min)
- Emphasize the **two-phase mental model**: define graph → run graph.
- Show the pseudocode: `placeholder`, `session.run`, `feed_dict`.
- Say: *"You can't print inside the graph. You don't know values until session.run()."*
- That's the bottleneck — debugging is indirect.

---

### Era 3 — Keras (3 min)
- `compile()` + `fit()` — most beginner-friendly ever.
- But: what's hidden? All of steps 3–5 from our 7 jobs.
- Today's Keras notebook uses this. The goal: see *through* it to what it's doing.

---

### Era 4 — Eager / Define-by-run (3 min)
- Show the `requires_grad` demo. Emphasize: **forward runs immediately**.
- `loss.backward()` traces back through the recorded ops.
- Ask: *"What's the difference between this and the graph-first era?"*

---

### Era 5 — PyTorch Mainstream (5 min) ← most important
- Walk through the full `TextClassifier` code slowly.
- Point to each of the 7 jobs in the code.
- Ask: *"Where is job 5 (backward pass)?"* → `loss.backward()`
- Ask: *"Where is job 4 (optimiser step)?"* → `opt.step()`
- Takeaway: **"Nothing is hidden. Everything is Python."**

---

### Era 6 — torch.compile (2 min)
- One line: `model = torch.compile(model)`
- Training loop unchanged. Just faster.
- Say: *"This is the direction — not a new thing to learn right now."*

---

## Section I — Keras ↔ PyTorch Comparison (10 min)

**Purpose:** Give learners a translation dictionary.

### Key talking points

1. **Concept mapping table** — scan it top to bottom. For each row say: *"Same idea, different syntax."*

2. **Decision matrix** — walk the rows. Ask: *"When would you choose Keras?"* and *"When would you choose PyTorch?"*

3. **Scenario cards** — pick 2–3 and ask the cohort to vote before you reveal the answer.

**Best scenario to discuss:** The "fine-tuning a HuggingFace model" card — this is their near-future workflow.

---

## Section J — Why PyTorch (8 min)

**Purpose:** Make the course's framework choice feel earned, not arbitrary.

### Key talking points

1. **4 cards** — read each headline and evidence cue. Don't over-explain.

2. **"Why it feels faster to iterate" diagram** — Idea → Code → Debug → Experiment → repeat. *"That tight loop is what builds intuition."*

3. **Gradient clipping snippet** — show the one-liner. Ask: *"How would you do this in Keras `fit()`?"* (You need a custom training step — more effort.)

4. **Debug comparison** — show the graph-first vs eager side-by-side. Ask them to find where `print()` works and where it doesn't.

**Key message:** *"PyTorch didn't win by being better at everything. It won by removing friction from the write → crash → fix loop."*

---

## Section K — Course Note (5 min)

**Purpose:** Connect today to what comes next.

### Key talking points

1. **Three columns** — Today (Keras scaffold) → Next module (PyTorch only) → Course arc (fine-tuning GenAI models)

2. **Outcomes checklist** — go through each item. Ask learners to self-assess silently.

3. **Translation exercise** — if time permits, spend 5 minutes having them attempt the PyTorch skeleton from the Keras given code. Don't need to complete it — just start.

**Closing line:**
> "From next module, everything is PyTorch. The concepts don't change — just you're now writing the loop yourself. And that's exactly the point."

---

## Q&A anchors (common questions)

| Question | Short answer |
|---|---|
| "Why does `CrossEntropyLoss` expect logits?" | It applies softmax internally — more numerically stable. Don't double-apply. |
| "What if I forget `model.eval()`?" | Dropout stays active → noisy eval metrics. BN uses batch stats instead of running stats. |
| "When do I use Keras vs PyTorch?" | Speed/prototyping → Keras. Custom loops, research, HuggingFace ecosystem → PyTorch. |
| "Is PyTorch always slower than TF?" | No — `torch.compile` closes most gaps. For typical fine-tuning workloads, difference is negligible. |
| "Why `.to(device)` on both model and data?" | They must be on the same device. Model params live on GPU; input tensors start on CPU. |

---

## Timing summary

| Section | Time |
|---|---|
| A + B (Hero + Agenda) | 3 min |
| C (NN Basics) | 10 min |
| D (7 Jobs) | 12 min |
| E (Training Loop) | 12 min |
| F (Why Python) | 8 min |
| G (Timeline) | 10 min |
| H (Era Cards × 6) | 20 min |
| I (Comparison) | 10 min |
| J (Why PyTorch) | 8 min |
| K (Course Note) | 5 min |
| Buffer / Q&A | 12 min |
| **Total** | **~110 min** |
