# MicroGrad architecture

This library has two layers.

## C++ core (`include/micrograd/`, `src/core/`)

The core provides:

- `Device` and `Stream` — a single dispatch path for the eventual CPU and
  CUDA backends. v0.1 only ships CPU.
- `Storage` — ref-counted per-device memory. CPU is `std::vector`-backed; CUDA
  will use a caching allocator.
- `Tensor` — a handle to a `Storage` + shape + dtype + device + a
  `requires_grad` flag + a pointer to the producing `IRNode` so the autograd
  engine can walk back.
- `Op` — a trait that carries `name`, `dtype`, `device`, a `forward` and a
  `backward` kernel. Each `(name, dtype, device)` triple gets its own
  registered `Op`; CPU and CUDA are two entries with the same name.
- `OpRegistry` — the dispatch table.
- `IRNode` / `Graph` — the captured forward. Backward is a single linear pass
  over `reverse_order()` with topological release.
- `Function` — a Python-facing callable that holds a `Graph` and exposes
  `apply`, `backward`, the four functional transforms, and `save`/`load`.
- `Module` — base for `Linear`, `Sequential`, `Conv2d`, `ReLU`, `Softmax`.
  `parameters()` walks the children.
- `Optimizer` — base for `SGD`, `Momentum`, `Adam`.
- Loss functions and the autograd engine (`backward`) live in `src/core/`.

## Python binding (`src/python/`, `python/micrograd/`)

pybind11 exposes the C++ classes. The Python `micrograd.tensor.Tensor` is a
thin wrapper that adds numpy interop, a `no_grad()` context manager, and a
`Function` cache for the autograd engine.

The user-facing functional transforms (`grad`, `vmap`, `jit`, `pmap`) live in
`python/micrograd/transforms/`. They consume and produce `Function` objects so
they compose: `vmap(grad(f))`, `pmap(jit(f))`, etc.

## How each of the seven flaws is defeated

1. **Eager/Graph duality is leaky.** Hybrid: eager forward records an IR
   into a `Graph`; backward consumes it as a single pre-planned pass.
2. **Backprop memory scales with loop depth.** The IR is reversed at build
   time; during backward, each input's `refcount` is decremented and its
   storage freed when it hits zero. Single linear pass.
3. **Custom op C++ boilerplate.** `@op_def` decorator lowers a Python
   forward+grad into a registered `Op` slot.
5. **Distributed API fragmentation.** `Mesh` + `ShardSpec`; `pmap` lowerer
   inserts collectives. (v0.2.)
6. **SavedModel fragility.** Versioned JSON-based graph + content-addressed
   tensor blob store. (v0.1 ships a simple JSON format; flatbuffer schema
   is in `schema/` for v0.2.)
7. **No first-class functional transforms.** `grad`, `vmap`, `jit`, `pmap`
   all consume and produce `Function` objects.
