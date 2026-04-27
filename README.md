# tilecraft

`tilecraft` is a hands-on GPU kernel practice codebase for learning and benchmarking tiled matrix computation with Triton and CuTe DSL.

The goal is to build kernels from simple reference implementations to progressively optimized versions, covering GEMM, attention, reductions, memory movement, layout transformations, and performance analysis.

## Environment

This repo is a study notebook and solution staging area for leetgpu-style kernel problems. In normal use, a single file such as `cute_dsl_kernels/vector_addition.py` is copied into the leetgpu editor/submission box.

Local macOS development is useful for editing, linting, type checking, and CPU-side reference code. Triton and CuTe DSL kernel compilation still belongs on Linux CUDA runners such as leetgpu.

Recommended baseline:

- Python 3.10-3.14
- Linux
- NVIDIA GPU with a compatible CUDA driver
- CUDA Toolkit 12.9 for `nvidia-cutlass-dsl`, or CUDA Toolkit 13.1 for `nvidia-cutlass-dsl[cu13]`

Create an environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev,triton,torch,cute-cu12]"
```

For local macOS editing, use the repository virtualenv and install only the packages that have macOS wheels:

```bash
/opt/homebrew/bin/python3.12 -m venv .venv
.venv/bin/python -m pip install -U pip setuptools wheel
.venv/bin/python -m pip install -r requirements/local-dev.txt
```

Triton and NVIDIA CuTe DSL are kept as Linux CUDA dependencies. VSCode uses the checked-in `typings/` stubs so imports such as `import cutlass.cute as cute` can still be analyzed locally.

Triton also has an interpreter mode:

```bash
TRITON_INTERPRET=1 python path/to/triton_test.py
```

In interpreter mode, Triton bypasses compilation and simulates kernels on CPU with NumPy equivalents. This is useful for checking indexing, masks, shapes, and many Python-level logic errors. It still requires the `triton` Python package to be importable, so it works in an environment where Triton can be installed. Current macOS arm64/Python 3.12 PyPI wheels are not available, so this project uses stubs locally and expects interpreter/runtime checks on a Linux environment.

For CUDA 13.1, install the CuTe DSL CUDA 13 extra instead:

```bash
python -m pip install -e ".[dev,triton,torch,cute-cu13]"
```

If leetgpu does not use editable installs, install the split requirements directly:

```bash
python -m pip install -r requirements/dev.txt
python -m pip install -r requirements/triton.txt
python -m pip install -r requirements/cute-cu12.txt
```

Use `requirements/cute-cu13.txt` instead of `requirements/cute-cu12.txt` on CUDA 13.1.

Check the environment:

```bash
python scripts/check_env.py
```

Official references:

- NVIDIA CUTLASS CuTe DSL quick start: <https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/quick_start.html>
- Triton installation guide: <https://triton-lang.org/main/getting-started/installation.html>

## Layout

- `cute_dsl_kernels/`: NVIDIA CuTe DSL practice kernels
- `triton_kernels/`: Triton practice kernels
- `tests/`: correctness tests
- `benchmarks/`: performance measurement scripts
- `notes/`: problem notes and optimization logs
