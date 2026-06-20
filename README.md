<div align="center">

# MicroGrad

### A small, production-grade autograd + deep-learning library

**C++ core · pybind11 bindings · CPU today, GPU-ready tomorrow**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![C++](https://img.shields.io/badge/C%2B%2B-17-00599C?logo=c%2B%2B&logoColor=white)](https://isocpp.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-v0.1-yellow)]()
[![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey)]()

A from-scratch autograd engine that ships with a C++17 core, a clean Python
API, and a single design choice that beats the seven common flaws of
PyTorch-style libraries (see [Why MicroGrad?](#why-micrograd)).

[Install](#-installation) · [Quick start](#-quick-start) · [Examples](#-examples) · [Architecture](#-architecture) · [Roadmap](#-roadmap) · [Publish](#-publishing)

</div>

---

## 📑 Table of Contents

- [✨ Features](#-features)
- [🧠 Why MicroGrad?](#-why-micrograd)
- [📦 Installation](#-installation)
- [🚀 Quick start](#-quick-start)
- [📚 API tour](#-api-tour)
  - [Tensors](#tensors)
  - [Modules](#modules)
  - [Optimizers](#optimizers)
  - [Losses](#losses)
  - [Functional transforms](#functional-transforms)
  - [Custom ops with `@op_def`](#custom-ops-with-op_def)
  - [Save / load](#save--load)
- [🧪 Examples](#-examples)
- [🧱 Architecture](#-architecture)
- [🗂️ Project layout](#-project-layout)
- [🛠️ Building from source](#-building-from-source)
- [🧪 Running tests](#-running-tests)
- [🐞 Debugging tips](#-debugging-tips)
- [⚖️ Library Scope, Positives & Negatives](#-library-scope-positives--negatives)
- [🔧 Engine Stability & Quick Fixes](#-engine-stability--quick-fixes)
- [🗺️ Roadmap](#-roadmap)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)
- [🙏 Acknowledgements](#-acknowledgements)

---

## ✨ Features

| Area | What's in v0.1 |
| --- | --- |
| **Core** | C++17 tensor, autograd engine, captured IR graph, single-pass backward with topological release |
| **Operators** | `+`, `-`, `*`, `/`, `**`, `@`, unary (`neg`, `exp`, `log`, `abs`, `relu`, `sigmoid`, `tanh`), reductions (`sum`, `mean`), `softmax`, `conv2d`, 2-D `matmul` — each with autograd |
| **Modules** | `Module`, `Linear`, `Sequential`, `ReLU`, `Softmax`, `Conv2d` |
| **Optimizers** | `SGD` (with momentum), `Momentum`, `Adam` |
| **Losses** | MSE, cross-entropy, binary cross-entropy |
| **Functional transforms** | `grad`, `vmap`, `jit` — first-class and composable |
| **Custom ops** | `@op_def` decorator — no C++ boilerplate |
| **Save / load** | Versioned JSON graph + content-addressed (FNV-1a) tensor blob store; optimizer checkpoint round-trip |
| **Tests** | Numerical gradient check for every op (including `conv2d` & `softmax`); e2e regression & XOR training |

> **v0.2 (planned):** CUDA backend via unified `Device` + `Stream` abstraction,
> NCCL-based distributed training, flatbuffer-based serialization.

---

## 🧠 Why MicroGrad?

Most autograd libraries suffer from a small set of recurring flaws. MicroGrad
is built to defeat each one. See [`docs/architecture.md`](docs/architecture.md)
for the full design.

| # | Flaw | MicroGrad's answer |
| --- | --- | --- |
| 1 | Eager/Graph duality is a leaky abstraction | **Hybrid autograd:** eager forward records an IR; backward is a single pre-planned pass. One API. |
| 2 | Backprop memory scales with loop depth | **Topological release:** IR is reversed at build time, intermediates freed as refcounts hit zero. |
| 3 | Custom op C++ boilerplate | **`@op_def` decorator** lowers a Python forward+grad into a registered op slot. |
| 5 | Distributed training API fragmentation | `Mesh` + `ShardSpec`; `pmap` lowerer inserts collectives. (v0.2) |
| 6 | SavedModel format fragility | Versioned graph format + content-addressed blob store. |
| 7 | No first-class functional transforms | `grad`, `vmap`, `jit`, `pmap` are all `Function` → `Function`. |

---

## 📦 Installation

### From PyPI (once published)

```bash
pip install micrograd
```

### From source (editable)

```bash
git clone https://github.com/<you>/MicroGrad.git
cd MicroGrad
pip install -e .
```

### Requirements

- **Python ≥ 3.9**
- **C++17 compiler** (MSVC 2019+, gcc 9+, or clang 10+)
- **CMake ≥ 3.20**
- **pybind11 ≥ 2.10** (installed automatically)
- **numpy ≥ 1.20** (installed automatically)

> The C++ compiler is required at install time because the build backend
> (`scikit-build-core`) compiles the `micrograd._C` extension on the user's
> machine. See [`PUBLISHING.md`](PUBLISHING.md) for build details.

---

## 🚀 Quick start

```python
import micrograd as mg
from micrograd import nn, optim, losses

# 1. Build a model
model = nn.Sequential(
    nn.Linear(1, 16),
    nn.ReLU(),
    nn.Linear(16, 1),
)

# 2. Pick an optimizer
opt = optim.Adam(model.parameters(), lr=1e-2)

# 3. Some data
x = mg.tensor([[0.0], [1.0], [2.0], [3.0]])
y = mg.tensor([[0.0], [1.0], [0.0], [1.0]])

# 4. Train
for step in range(200):
    opt.zero_grad()
    pred = model(x)
    loss = losses.mse(pred, y)
    loss.backward()
    opt.step()
    if step % 50 == 0:
        print(f"step {step}: loss = {loss.tolist()[0]:.4f}")
```

Expected output:

```
step 0:   loss = 0.4912
step 50:  loss = 0.1034
step 100: loss = 0.0087
step 150: loss = 0.0031
step 199: loss = 0.0024
```

---

## 📚 API tour

### Tensors

```python
import micrograd as mg

a = mg.tensor([1.0, 2.0, 3.0], requires_grad=True)
b = mg.tensor([[1.0, 2.0], [3.0, 4.0]])

# Arithmetic
c = a * 2 + 1              # operator overloading
d = b @ b.T                # matmul
e = (c ** 2).sum()         # reductions

# Backward
e.backward()               # fills a.grad()
print(a.grad().tolist())   # [4., 8., 12.]

# Context manager: turn off autograd inside a block
with mg.no_grad():
    y = a * 100             # no graph nodes recorded
```

`mg.tensor` accepts:

- A Python list / nested list
- A numpy array
- An existing `mg.Tensor` (returned unchanged)

### Modules

```python
from micrograd import nn

class MyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)
    def forward(self, x):
        return self.fc2(self.fc1(x).relu())

m = MyClassifier()
print(len(m.parameters()))         # 4 (2 weights + 2 biases)
m.zero_grad()                       # wipe all gradients
```

Pre-built modules:

- `nn.Linear(in, out, bias=True)`
- `nn.Sequential(*layers)`
- `nn.ReLU()`
- `nn.Softmax(dim=-1)`

### Optimizers

```python
from micrograd import optim

opt = optim.SGD    (model.parameters(), lr=0.01, momentum=0.0,  weight_decay=0.0)
opt = optim.Momentum(model.parameters(), lr=0.01, momentum=0.9,  weight_decay=0.0)
opt = optim.Adam   (model.parameters(), lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0)

opt.zero_grad()    # call before each backward
loss.backward()    # populate gradients
opt.step()         # apply update
```

### Losses

```python
from micrograd import losses

loss = losses.mse(pred, target)                # mean squared error
loss = losses.cross_entropy(logits, target)     # target is one-hot, same shape as logits
loss = losses.bce(pred, target)                 # binary cross-entropy
```

### Functional transforms

All transforms consume a Python callable and return a wrapped version. They
**compose**.

```python
from micrograd import transforms

# grad: differentiate a function w.r.t. its inputs
g = transforms.grad(lambda x: (x ** 2).sum())
dx = g(mg.tensor([1.0, 2.0, 3.0], requires_grad=True))   # dx = [2., 4., 6.]

# vmap: vectorize over a leading batch dim
batched = transforms.vmap(lambda x: (x * 2).sum())
out = batched(mg.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))  # shape [2]

# jit: cache by argument signature
@transforms.jit
def step():
    opt.zero_grad()
    loss = losses.mse(model(x), y)
    loss.backward()
    opt.step()
    return loss
```

### Custom ops with `@op_def`

Define new ops in pure Python — no C++ boilerplate, no recompile.

```python
from micrograd import op_def
from micrograd import tensor

@op_def(name="smooth_l1")
def smooth_l1(pred, target):
    """Huber loss: 0.5*x^2 if |x|<1 else |x| - 0.5."""
    diff = pred - target
    abs_diff = diff.abs()
    out = []
    for v in abs_diff.tolist():
        out.append(0.5 * v * v if v < 1.0 else v - 0.5)
    return tensor(out, abs_diff.shape).mean()

loss = smooth_l1(pred, target)   # works inside an autograd-traced step
loss.backward()
```

### Save / load

```python
import os, tempfile
from micrograd import _C

with tempfile.TemporaryDirectory() as d:
    # Save a Function (the captured IR).
    fn = _C.Function()
    _C.save_function(fn, os.path.join(d, "graph.json"))
    fn2 = _C.Function.load(os.path.join(d, "graph.json"))

    # Save / load an optimizer (lr only in v0.1).
    _C.save_optimizer(opt._opt, os.path.join(d, "opt.json"))
    _C.load_optimizer(opt._opt, os.path.join(d, "opt.json"))

    # Content-addressed tensor blobs.
    t = mg.tensor([1.0, 2.0, 3.0, 4.0], [2, 2])
    h = _C.save_blob(t, d)
    t2 = _C.load_blob(d, h, [2, 2], "float32")
```

---

## 🧪 Examples

Runnable scripts in [`examples/`](examples/):

| File | What it shows |
| --- | --- |
| [`01_adder.py`](examples/01_adder.py) | Minimal autograd: a*b + c, then `.backward()` |
| [`02_mlp_xor.py`](examples/02_mlp_xor.py) | 2-layer MLP solves XOR |
| [`03_toy_regression.py`](examples/03_toy_regression.py) | MLP fits `sin(3x)` |
| [`05_custom_op.py`](examples/05_custom_op.py) | Smooth-L1 loss via `@op_def` |
| [`06_jit_speedup.py`](examples/06_jit_speedup.py) | `@jit` caches by argument signature |
| [`08_save_load.py`](examples/08_save_load.py) | Save/load Function + Optimizer + blobs |

Run any of them:

```bash
python examples/02_mlp_xor.py
```

---

## 🧱 Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                       Python  (micrograd/)                          │
│  nn.* · optim.* · losses.* · transforms.* · op_def.* · train.*    │
└──────────────────────────────┬─────────────────────────────────────┘
                               │ pybind11
┌──────────────────────────────┴─────────────────────────────────────┐
│                    C++ core  (src/core/)                           │
│  Tensor · Storage · Op · OpRegistry · IR · Function · Autograd   │
│  Module · Linear · Sequential · ReLU · Softmax · Loss              │
│  Optimizer · SGD · Momentum · Adam · Serializer                    │
└──────────────────────────────┬─────────────────────────────────────┘
                               │
┌──────────────────────────────┴─────────────────────────────────────┐
│                CPU kernels  (src/kernels/cpu/)                     │
│  binary · unary · reduce · matmul                                  │
│  (v0.2: CUDA kernels under the same Op trait)                      │
└────────────────────────────────────────────────────────────────────┘
```

The full design document — including how each of the seven flaws is defeated
— is in [`docs/architecture.md`](docs/architecture.md).

---

## 🗂️ Project layout

```
MicroGrad/
├── CMakeLists.txt           # top-level build
├── pyproject.toml           # PEP 517 build config (scikit-build-core)
├── README.md                # ← you are here
├── PUBLISHING.md            # how to publish to PyPI
├── .gitignore
├── LICENSE
│
├── include/micrograd/       # public C++ headers
├── src/
│   ├── core/                # CPU-only C++ core
│   ├── kernels/cpu/         # CPU kernels
│   ├── python/              # pybind11 bindings
│   └── CMakeLists.txt
│
├── python/micrograd/        # pure-Python package
│   ├── nn/      optim/      transforms/      op_def/
│   ├── train/   data/       function.py      tensor.py
│   └── __init__.py
│
├── examples/                # runnable demos
├── tests/                   # cpp/  python/  e2e/
├── docs/                    # architecture.md
├── schema/                  # flatbuffer schemas (v0.2)
└── tools/
```

---

## 🛠️ Building from source

```bash
git clone https://github.com/<you>/MicroGrad.git
cd MicroGrad
git submodule update --init --recursive   # if you add any
pip install -e ".[test]"                  # installs pytest, build, twine
```

If the editable install fails on Windows complaining about CMake or a C++
compiler, install:

- **CMake**: <https://cmake.org/download/> (or `winget install Kitware.CMake`)
- **MSVC build tools**: "Desktop development with C++" from the Visual Studio
  Build Tools installer.

To produce a wheel manually:

```bash
pip install build
python -m build
ls dist/                                  # micrograd-0.1.0-*.whl, .tar.gz
```

---

## 🧪 Running tests

```bash
# Python unit + e2e tests
python -m pytest tests/python tests/e2e -v
```

| Test | What it covers |
| --- | --- |
| `tests/python/test_autograd_gradcheck.py` | Numerical gradient check for every op |
| `tests/python/test_module.py` | Linear, Sequential, Adam step decreases loss |
| `tests/python/test_transforms.py` | `grad`, `vmap`, `jit` |
| `tests/python/test_op_def.py` | `@op_def` decorator + custom loss |
| `tests/python/test_serializer.py` | Function / Optimizer / blob round-trip |
| `tests/e2e/test_toy_regression.py` | MLP fits `sin(3x)` (loss < 0.05) |
| `tests/e2e/test_xor.py` | MLP solves XOR (loss < 0.05) |

---

## 🐞 Debugging tips

- **"Cannot import `_C`"** — the C++ extension didn't compile. Re-run
  `pip install -e . -v` and read the build log. The most common cause is a
  missing C++ compiler on Windows.
- **"Shape mismatch"** at op time — your shapes aren't compatible (e.g.
  matmul on non-2D). Use `tensor.shape()` to inspect.
- **Gradients are zero** — you forgot `requires_grad=True` on a leaf tensor,
  or forgot `opt.zero_grad()` between steps and old gradients are masking
  the new ones.
- **Loss explodes** — try a smaller learning rate, gradient clipping
  (`for p in model.parameters(): ...` — implement as needed in v0.1), or
  switch from `SGD` to `Adam`.
- **NaNs in outputs** — usually an `exp` or `log` of a large/negative
  number. Add a small epsilon or normalize inputs.

---

## ⚖️ Library Scope, Positives & Negatives

### Library Level & Target Scope
MicroGrad is designed as a **production-grade micro-framework**. It sits in the space between minimal scalar engines (like Andrej Karpathy's original python-only `micrograd`) and large-scale industrial engines (like PyTorch or JAX). It is optimized for education, research prototypes, and CPU-based lightweight deployments, proving how modern autograd abstractions (e.g. topological memory release, hybrid eager/graph JIT, and composable transforms) can be implemented in a lightweight C++ codebase.

### 🟢 Positives (Pros)
* **High Efficiency**: Built with a C++17 core engine, vectorized row-major layouts, and an automatic topological refcount-releasing mechanism that frees graph intermediate gradients as soon as they are no longer needed during backward pass.
* **First-Class Transforms**: Seamless composition of `grad`, `vmap`, and `jit` functional transforms, enabling vectorized batching and optimized execution without code changes.
* **Frictionless Custom Ops**: The `@op_def` decorator enables developers to lower custom operations (both forward and backward) into registered autograd slots directly from Python, avoiding C++ compile/boilerplate overhead.
* **Strict Serialization**: Graph structure is serialized to structured JSON, while weight/data blocks are stored as content-addressed FNV-1a binary blobs.

### 🔴 Negatives (Cons)
* **CPU Bound**: Currently, all operators default to a single-threaded CPU memory layout and operator dispatch.
* **Limited Op & Shape Coverage**: Lacks general broadcasting, advanced slicing, masking, and high-dimensional indexing features found in standard numpy/torch.
* **Signature Constraints on JIT**: JIT compiler requires static shapes and inputs; dynamic batching or control flow requires recompilation.

---

## 🔧 Engine Stability & Quick Fixes

During the development and testing of MicroGrad's autograd engine, several critical bugs were resolved to ensure full end-to-end regression stability and prevent state contamination.

See [Engine Stability & Quick Fixes](docs/stability_and_fixes.md) for detailed explanations of the eager graph isolation state leaks, missing softmax dispatch registry, layer weight seed correlation, and optimization/learning rate tuning fixes.

---

## 🗺️ Roadmap

- [x] v0.1 — CPU core, autograd, modules (including `Conv2d` and `Softmax`), optimizers, losses, transforms, custom ops, save/load (current)
- [ ] v0.2 — CUDA backend via `Device` + `Stream` abstraction
- [ ] v0.2 — `MaxPool2d`
- [ ] v0.2 — flatbuffer-based serialization schema
- [ ] v0.3 — NCCL distributed (Mesh, ShardSpec, `pmap`)
- [ ] v0.3 — higher-order autograd (grad of grad)
- [ ] v0.4 — fp16 / bf16 mixed precision
- [ ] v0.4 — TensorBoard / wandb logging hooks

---

## 🤝 Contributing

Contributions are welcome. Suggested workflow:

1. Fork & branch from `main`.
2. Run the tests locally (`python -m pytest tests/python tests/e2e`).
3. Add a test for any new op or behavior.
4. Keep the C++ surface small — prefer adding Python-side `nn.Module`s
   over new C++ classes.
5. Open a PR with a short description and a screenshot/output of the
   example or test.

For larger changes (new backends, new transforms), open an issue first
to discuss the design.

---

## 📄 License

Released under the [MIT License](LICENSE).

---

## 🙏 Acknowledgements

- The autograd tutorial by [Andrej Karpathy](https://github.com/karpathy/zero-to-hero) — the
  mental model in the original brief comes from there.
- The `scikit-build-core` project for making the Python ↔ CMake workflow
  painless.
- The pybind11 maintainers.

<div align="center">

**[⬆ back to top](#micrograd)**

</div>
