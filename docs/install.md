# Installation

## From PyPI

EvoFlow is published as **`evoflow-bio`**; the import name and console script
are `EvoFlow` and `evoflow`.

```bash
pip install evoflow-bio
```

Python 3.10–3.12 is supported.

## Optional extras

| Extra       | Contents                                   | Use case                          |
| ----------- | ------------------------------------------ | --------------------------------- |
| `docs`      | `sphinx`, `furo`, `myst-nb`, `copybutton`  | Build these documentation pages   |
| `vignette`  | `pertpy`, `nichenetpy`                     | Run the use-case notebook         |
| `dev`       | `ruff`, `pytest`, `pytest-mock`, `pre-commit` | Development and tests          |

```bash
pip install "evoflow-bio[docs]"
pip install "evoflow-bio[vignette]"
pip install "evoflow-bio[dev]"
```

## From source

```bash
git clone https://github.com/VergaJU/EvoFlow.git
cd EvoFlow
pip install -e ".[dev]"
pytest
```

## Docker

If you also need NicheNet inference and the example single-cell dataset, use
the published container (or install the `vignette` extra, which pulls in
`nichenetpy` and `pertpy`). See {doc}`container`.

## Building the documentation locally

```bash
pip install ".[docs]"
sphinx-build -W -b html docs docs/_build/html
```

The rendered site is written to `docs/_build/html`. Notebook outputs are
rendered as committed (`nb_execution_mode = "off"`), so no R or Docker is
required to build the docs.
