# Publishing MicroGrad to PyPI

This guide walks through building and publishing the `micrograd` package to
[PyPI](https://pypi.org/). The project uses
[`scikit-build-core`](https://scikit-build-core.readthedocs.io/) as its
build backend, so the standard `python -m build` + `twine` workflow works
without any extra glue.

> **Heads up on the package name.** PyPI package names are
> case-insensitive and global. Before publishing, search
> <https://pypi.org/search/?q=micrograd> and pick an available name (or
> scope under a prefix like `micrograd-cuda`). Update
> `pyproject.toml` (`[project].name`) and the wheel output name
> (`OUTPUT_NAME` in `src/CMakeLists.txt`) accordingly.

---

## 1. Prerequisites

You need:

| Tool | Min version | Install |
| --- | --- | --- |
| Python | 3.9 | <https://python.org/downloads/> |
| `pip` | 23+ | bundled with Python |
| `cmake` | 3.20+ | <https://cmake.org/download/> |
| C++ compiler | C++17 | MSVC 2019+, gcc 9+, or clang 10+ |
| `build` | latest | `pip install build` |
| `twine` | latest | `pip install twine` |

A C++ compiler is required because `scikit-build-core` invokes `cmake` to
build the `micrograd._C` extension on the user's machine.

---

## 2. One-time PyPI setup

1. Create a PyPI account at <https://pypi.org/account/register/>.
2. Enable two-factor authentication (required by PyPI).
3. Create an API token at <https://pypi.org/manage/account/token/>.
   Scope the token to the project (or to "Entire account" for the first
   upload).
4. (Optional but recommended for CI) configure
   [trusted publishing](https://docs.pypi.org/trusted-publishers/) from
   GitHub Actions so you don't need to manage long-lived tokens.

For TestPyPI (recommended dry-run target), repeat the steps at
<https://test.pypi.org/>.

---

## 3. Configure credentials locally

Put your token in `~/.pypirc`:

```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = pypi-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
```

`twine` reads this file. Do **not** commit it. The `.gitignore` already
excludes it via the leading dotfile rules.

You can also pass `--username __token__ --password pypi-...` on the
command line, or use `keyring` (`pip install keyring && keyring set
https://upload.pypi.org/legacy/ __token__`).

---

## 4. Build the wheel and sdist

From the repository root:

```bash
# Clean any previous build
rm -rf build/ dist/ *.egg-info

# Build sdist + wheel
python -m build
```

This runs `scikit-build-core`, which:

1. Compiles the C++ extension via `CMakeLists.txt` → produces
   `micrograd/_C.<plat>.so` (Linux/macOS) or `_C.cp<python>-<plat>.pyd`
   (Windows).
2. Bundles `python/micrograd/*.py` and the compiled extension into a
   wheel (`dist/micrograd-0.1.0-*.whl`).
3. Produces a source distribution (`dist/micrograd-0.1.0.tar.gz`).

Inspect the wheel to verify the layout:

```bash
unzip -l dist/micrograd-0.1.0-*.whl
```

You should see `micrograd/__init__.py`, `micrograd/_C.*.so` (or `.pyd`),
and the rest of the `python/micrograd/` tree.

---

## 5. Smoke-test the wheel locally

```bash
python -m venv .venv-test
. .venv-test/bin/activate    # Windows: .venv-test\Scripts\activate
pip install dist/micrograd-0.1.0-*.whl
python -c "import micrograd; print(micrograd.__version__)"
python examples/01_adder.py
```

If the import works and the example prints the expected output, the
wheel is good to publish.

---

## 6. Publish to TestPyPI first (highly recommended)

```bash
python -m twine upload --repository testpypi dist/*
```

This pushes to <https://test.pypi.org/project/micrograd/>. Test the
install from there in a clean environment:

```bash
python -m venv .venv-testpypi
. .venv-testpypi/bin/activate
pip install --index-url https://test.pypi.org/simple/ \
            --extra-index-url https://pypi.org/simple/ \
            micrograd==0.1.0
python -c "import micrograd; print(micrograd.__version__)"
```

(The `--extra-index-url` lets TestPyPI fall back to real PyPI for
dependencies like `numpy`.)

---

## 7. Publish to PyPI

```bash
python -m twine upload dist/*
```

You should see output ending in:

```
Uploading micrograd-0.1.0-*.whl
Uploading micrograd-0.1.0.tar.gz
View at:
https://pypi.org/project/micrograd/0.1.0/
```

---

## 8. Versioning workflow

This project follows [Semantic Versioning](https://semver.org/):
`MAJOR.MINOR.PATCH`.

For a new release:

1. Edit `python/micrograd/__init__.py` and bump `__version__`.
2. Edit `pyproject.toml` and bump `[project].version` to match.
3. Commit the bump:

   ```bash
   git add python/micrograd/__init__.py pyproject.toml
   git commit -m "Release 0.1.0"
   ```

4. Tag the release:

   ```bash
   git tag -a v0.1.0 -m "Release 0.1.0"
   git push origin main --tags
   ```

5. Rebuild and re-publish:

   ```bash
   rm -rf build/ dist/
   python -m build
   python -m twine upload dist/*
   ```

PyPI does not allow re-uploading the same version. To fix a release,
bump to `0.1.1` and re-publish.

---

## 9. Post-publish verification

```bash
python -m venv .venv-check
. .venv-check/bin/activate
pip install --upgrade micrograd
python -c "import micrograd; print(micrograd.__version__)"
python examples/01_adder.py
```

---

## 10. Common issues

### "File already exists"
PyPI rejects duplicate uploads. Bump the version in both
`python/micrograd/__init__.py` and `pyproject.toml`, then rebuild.

### "Invalid distribution filename"
The wheel name and Python version tags must match what the build
backend produces. `python -m build` should do the right thing; if you
renamed the project, check `[project].name` in `pyproject.toml`.

### "CMake Error: CMAKE_CXX_COMPILER not set"
Install a C++ compiler (see Prerequisites) and re-run `python -m build`.

### "ModuleNotFoundError: No module named 'micrograd._C' after install"
The C++ extension failed to compile. Re-run
`python -m build -v` and check the build log. On Windows you typically
need "Desktop development with C++" from the Visual Studio Build Tools
installer.

### "ImportError: DLL load failed while importing _C" (Windows)
The Python version used to build the wheel doesn't match the
interpreter trying to load it. Build with the same Python (and same
architecture, e.g. CPython 3.11 x64) you intend to install into.

### "Long description has invalid syntax"
If you add a long description, make sure the file referenced by
`[project.readme]` in `pyproject.toml` is valid Markdown or reStructuredText
and is committed to the repo (or to the sdist).

---

## 11. CI publishing (optional)

For trusted publishing from GitHub Actions, add `.github/workflows/publish.yml`:

```yaml
name: publish
on:
  push:
    tags: ['v*']
jobs:
  build-and-publish:
    runs-on: ubuntu-latest
    permissions:
      id-token: write   # required for trusted publishing
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - name: Install build deps
        run: |
          pip install build
          sudo apt-get update
          sudo apt-get install -y cmake g++
      - name: Build
        run: python -m build
      - name: Publish to PyPI
        uses: pypa/gh-action-pypi-publish@release/v1
        with:
          packages-dir: dist/
```

Configure trusted publishing on PyPI:
<https://docs.pypi.org/trusted-publishers/adding-a-publisher/>.

---

## 12. License reminder

`pyproject.toml` should declare a license. If you haven't already, add
SPDX metadata and either a `LICENSE` file at the repo root or a
`license = {text = "MIT"}` entry under `[project]`.
