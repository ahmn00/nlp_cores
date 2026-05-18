# The PyTorch Training Workflow — Deep Dive

> **Goal:** Understand every step of defining a model, configuring an optimiser, evaluating, and running a training loop — so well that you could rebuild it from scratch.

---

## Mental Model: The Big Picture

Before touching any code, lock this mental model in your head:

```
Raw Data
   │
   ▼
[Model] ──── forward pass ────► Predictions (logits)
                                        │
                              [Loss Function] ◄── True Labels
                                        │
                                    Loss value
                                        │
                              [.backward()] ── computes gradients
                                        │
                              [Optimiser.step()] ── updates weights
                                        │
                                 Repeat until converged
```

Training is nothing more than this loop repeated thousands of times. Every concept below fits into one of these boxes.

---

## Part 1 — Defining the Model Class (`nn.Module`)

### Why a class?

A PyTorch model is a Python class that inherits from `nn.Module`. This gives you:
- Automatic tracking of all learnable parameters (weights, biases)
- A standard interface (`model(x)`) that hooks into PyTorch's engine
- Utilities like `.parameters()`, `.to(device)`, `.train()`, `.eval()`, `.state_dict()`

### The two mandatory methods

```python
class MyModel(nn.Module):
    def __init__(self, ...):
        super().__init__()         # ALWAYS first — registers the module with PyTorch
        # define layers here

    def forward(self, x):
        # describe the computation here
        return output
```

**`__init__`** — the constructor. You define *what layers exist*. Nothing is computed here.

**`forward`** — describes *how data flows through those layers*. This is called every time you do `model(x)`.

> ⚠️ **`forward` is mandatory and cannot be renamed.** PyTorch's `nn.Module.__call__` is hardcoded to call `self.forward(...)`. If you rename it, `model(x)` silently falls back to the base class stub which raises `NotImplementedError`.

### Example: building up intuitively

```python
# Step 1: a useless model that does nothing
class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

# Step 2: a single linear layer
class OneLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.fc(x)

# Step 3: two layers with an activation in between
class TwoLayer(nn.Module):
    def __init__(self, in_f, hidden_f, out_f):
        super().__init__()
        self.fc1 = nn.Linear(in_f, hidden_f)
        self.fc2 = nn.Linear(hidden_f, out_f)

    def forward(self, x):
        h = torch.relu(self.fc1(x))   # activation applied manually
        return self.fc2(h)
```

### What `nn.Linear` actually does

`nn.Linear(in, out)` creates a weight matrix `W` of shape `[out, in]` and a bias `b` of shape `[out]`. When called, it computes:

```
output = x @ W.T + b
```

That's it. All the "magic" is just matrix multiplication.

### What does `super().__init__()` do?

It calls `nn.Module.__init__()` which sets up internal data structures that PyTorch uses to:
- Track sub-modules (any `nn.Module` you assign to `self.something`)
- Track parameters (any `nn.Parameter` you assign)
- Enable `.parameters()`, `.to(device)`, etc.

Without it, PyTorch won't know your layers exist and `.parameters()` will return nothing — your model won't train.

---

## Part 2 — The Loss Function

### What is loss?

Loss (also called cost or objective) is a **single number** that measures how wrong the model's predictions are. The training loop's entire job is to minimise this number.

### CrossEntropyLoss — the standard for classification

```python
criterion = nn.CrossEntropyLoss()
loss = criterion(logits, labels)
```

**Inputs:**
- `logits`: raw model outputs, shape `[batch_size, num_classes]` — NOT probabilities yet
- `labels`: ground truth class indices, shape `[batch_size]`, dtype `int64`

**What it does internally:**
```
logits → Softmax → log → NLL Loss → scalar
```

Softmax converts raw scores into a probability distribution (sums to 1):
```
p_i = exp(logit_i) / Σ exp(logit_j)
```

Cross-entropy then measures how surprised the model is by the true label:
```
loss = -log(p_true_class)
```

If the model is very confident and correct → `p_true_class ≈ 1.0` → `loss ≈ 0`  
If the model is very confident and wrong → `p_true_class ≈ 0.0` → `loss → ∞`

**Why does `CrossEntropyLoss` accept raw logits instead of probabilities?**  
Numerical stability. Computing `log(softmax(x))` separately can produce `log(0) = -inf`. PyTorch's `log_softmax` is implemented with the "log-sum-exp" trick to avoid this.

### A concrete example

```python
# Model predicts for 3 classes, batch size 2
logits = torch.tensor([[2.0, 1.0, 0.1],   # sample 1: most confident about class 0
                        [0.5, 2.5, 0.3]])  # sample 2: most confident about class 1
labels = torch.tensor([0, 1])              # true labels

loss = nn.CrossEntropyLoss()(logits, labels)
# Both predictions are correct and fairly confident → low loss
```

---

## Part 3 — The Optimiser

### What is an optimiser?

After the loss is computed and `.backward()` computes gradients, the optimiser uses those gradients to update the model's weights. Without an optimiser, you'd have to manually update every parameter.

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
```

`model.parameters()` yields every learnable tensor in the model. The optimiser holds a reference to all of them and updates them in-place when `.step()` is called.

### SGD — the conceptual baseline

The simplest optimiser is Stochastic Gradient Descent:

```
w ← w - lr × ∇L(w)
```

Move each weight a small step (`lr`) in the direction that reduces the loss (`-∇L`).

### Adam — adaptive learning rates

Real loss surfaces are curved differently in every direction. A fixed `lr` for every parameter is inefficient. Adam maintains a **per-parameter learning rate** by tracking:

- `m` — exponential moving average of gradients (first moment / "momentum")
- `v` — exponential moving average of squared gradients (second moment / "variance")

```
m ← β₁ × m + (1 - β₁) × g          # smoothed gradient
v ← β₂ × v + (1 - β₂) × g²         # smoothed squared gradient
w ← w - lr × m / (√v + ε)           # adaptive update
```

Parameters with consistently large gradients get a smaller effective learning rate (they're on steep slopes). Parameters with small gradients get a larger effective learning rate (they're on flat slopes). This makes Adam much faster than SGD in practice.

### AdamW — fixing weight decay

Standard Adam applies weight decay **inside the adaptive scaling**, which interacts incorrectly with the momentum terms. AdamW decouples weight decay:

```
w ← w - lr × m / (√v + ε)   # Adam update (unchanged)
w ← w × (1 - lr × λ)         # weight decay applied separately
```

This makes regularisation behave correctly regardless of gradient magnitudes. **AdamW is now the standard default in modern deep learning.**

### The three-line update pattern

Every training step follows exactly this pattern:

```python
optimizer.zero_grad()    # 1. clear old gradients (they accumulate by default!)
loss.backward()          # 2. compute new gradients via backpropagation
optimizer.step()         # 3. update weights using those gradients
```

**Why `zero_grad()` first?**  
PyTorch *accumulates* gradients by default (adds to existing values). This is useful for some advanced techniques (gradient accumulation over multiple batches), but for standard training you must clear them before each backward pass or gradients from the previous batch corrupt the current one.

---

## Part 4 — Gradient Clipping

```python
nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

Called **after** `loss.backward()` but **before** `optimizer.step()`.

### Why is this needed?

During backpropagation through time (BPTT) in RNNs and LSTMs, gradients are multiplied across every time step. If the weight matrix has eigenvalues > 1, this multiplication causes gradients to grow exponentially — **exploding gradients**.

One bad batch can produce a gradient update so large that it pushes weights to extreme values, from which the model never recovers.

Gradient clipping caps the global L2 norm of all gradients:
```
if ||g|| > max_norm:
    g ← g × (max_norm / ||g||)
```

This preserves the *direction* of the gradient but limits its *magnitude*.

**Rule of thumb:** use `max_norm=1.0` for recurrent models, `max_norm=5.0` is also common. Transformer training is less sensitive but clipping is still good practice.

---

## Part 5 — The Evaluation Function

```python
def evaluate(model, loader, criterion):
    model.eval()
    with torch.no_grad():
        ...
    return avg_loss, accuracy
```

### `model.eval()` — what does it change?

Calling `.eval()` switches two types of layers into inference mode:

| Layer | Train mode | Eval mode |
|---|---|---|
| `nn.Dropout` | Randomly zeros some activations | Pass-through (no dropout) |
| `nn.BatchNorm` | Uses batch statistics | Uses running statistics |

You must call `.eval()` before any inference/evaluation to get correct results. After evaluation, call `model.train()` to switch back before the next training epoch.

### `torch.no_grad()` — why?

During training, PyTorch builds a **computational graph** alongside every operation so it can compute gradients via backpropagation. This graph takes memory and time to construct.

During evaluation you don't need gradients — you're only doing forward passes. `torch.no_grad()` disables graph construction, which:
- Reduces memory usage (no need to store intermediate activations for backprop)
- Speeds up computation (no graph overhead)

```python
# Without no_grad: builds graph, uses ~2x memory
logits = model(xb)

# With no_grad: no graph, faster and lighter
with torch.no_grad():
    logits = model(xb)
```

---

## Part 6 — The Training Loop (The Full Picture)

```python
for epoch in range(1, cfg.epochs + 1):
    model.train()

    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)

        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    # Evaluate at end of each epoch
    val_loss, val_acc = evaluate(model, val_loader, criterion)
```

### Annotated line-by-line

```python
for epoch in range(1, cfg.epochs + 1):
```
One epoch = one full pass over the entire training set. Multiple epochs let the model see the same data repeatedly and refine its weights.

```python
    model.train()
```
Switch to training mode (enables dropout, batch norm in train mode).

```python
    for xb, yb in train_loader:
```
`DataLoader` yields mini-batches. Each `xb` is shape `[batch_size, seq_len]`, each `yb` is shape `[batch_size]`. Mini-batches give a noisy but fast estimate of the true gradient (stochastic gradient descent).

```python
        xb, yb = xb.to(device), yb.to(device)
```
Move data to the same device as the model (CPU or GPU). This is mandatory — PyTorch raises an error if tensors are on different devices.

```python
        optimizer.zero_grad()
```
Clear gradients accumulated from the previous step.

```python
        logits = model(xb)
```
Forward pass: calls `model.__call__(xb)` which calls `model.forward(xb)` plus any hooks. `logits` has shape `[batch_size, num_classes]`.

```python
        loss = criterion(logits, yb)
```
Compute the scalar loss value for this batch.

```python
        loss.backward()
```
Backpropagation: PyTorch traverses the computational graph in reverse, computing `∂loss/∂w` for every learnable parameter `w`. These gradients are stored in `w.grad`.

```python
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```
Cap gradient magnitudes before applying them.

```python
        optimizer.step()
```
For each parameter `w`, apply the update rule (AdamW in our case) using `w.grad`.

---

## Part 7 — Why Mini-Batches?

You might wonder: why not compute the gradient over the *entire* dataset every step?

| Approach | Gradient quality | Speed | Memory |
|---|---|---|---|
| Full-batch GD | Exact | Slow (one update per epoch) | High |
| SGD (1 sample) | Very noisy | Fast | Low |
| **Mini-batch SGD** | Good estimate | Fast | Moderate |

Mini-batches hit the sweet spot. The noise from using a subset of data actually *helps* escape sharp local minima (a regularisation effect). Typical batch sizes: 32, 64, 128, 256.

---

## Part 8 — The History Dictionary Pattern

```python
history.append({
    'epoch': epoch,
    'train_loss': tr_loss,
    'train_acc': tr_acc,
    'val_loss': va_loss,
    'val_acc': va_acc
})
```

Logging metrics every epoch into a list of dicts is a lightweight but powerful pattern. Wrapping it in a DataFrame at the end gives you:

```python
hist_df = pd.DataFrame(history)
hist_df.plot(x='epoch', y=['train_acc', 'val_acc'])
```

This gives you your **learning curves** — the single most important diagnostic in training. A growing gap between train and val curves → overfitting. Both low and stuck → underfitting.

---

## Part 9 — Common Mistakes & Fixes

| Mistake | Symptom | Fix |
|---|---|---|
| Forgot `zero_grad()` | Loss oscillates or diverges | Add `optimizer.zero_grad()` at start of each batch |
| Forgot `model.eval()` | Val accuracy too low (dropout active) | Always call `model.eval()` before evaluating |
| Forgot `model.train()` after eval | Training accuracy stays low (dropout disabled) | Call `model.train()` at start of each epoch |
| Forgot `.to(device)` on data | RuntimeError: different devices | Move `xb, yb` to `device` each batch |
| Calling `model.forward(x)` directly | Hooks don't fire, subtle bugs | Always call `model(x)` not `model.forward(x)` |
| Using `loss.item()` without calling it in the loop | Keeps computational graph alive, OOM | Always call `.item()` to detach from graph |

---

## Part 10 — Putting It All Together: Minimal Working Example

Here is the full workflow in ~40 lines, completely from scratch:

```python
import torch
import torch.nn as nn

# 1. Define model
class TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.net(x)

# 2. Fake data
X = torch.randn(200, 10)
y = torch.randint(0, 2, (200,))
dataset = torch.utils.data.TensorDataset(X, y)
loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

# 3. Instantiate model, loss, optimiser
model = TinyClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# 4. Training loop
for epoch in range(10):
    model.train()
    for xb, yb in loader:
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()

    # 5. Evaluate
    model.eval()
    with torch.no_grad():
        val_logits = model(X)
        acc = (val_logits.argmax(dim=1) == y).float().mean()
    print(f"Epoch {epoch+1} | Accuracy: {acc:.3f}")
```

Every single training script in the world — from this toy example to GPT-4 — follows this exact same structure.
