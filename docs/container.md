# Container

A reproducible container with **R**, **NicheNet**, **JupyterLab** and
**EvoFlow** is published on the GitHub Container Registry:

```text
ghcr.io/vergaju/evoflow
```

It is built from a pinned date tag of
`quay.io/jupyter/datascience-notebook`, so the R and Python stacks are
reproducible.

## Pull and run

```bash
docker run --rm -p 8888:8888 ghcr.io/vergaju/evoflow:latest
```

Then open <http://localhost:8888> and use the token printed by JupyterLab.

To pin the release that matches these docs:

```bash
docker run --rm -p 8888:8888 ghcr.io/vergaju/evoflow:v1.0.0
```

Available tags are `latest`, `1.0.0`, `1.0`, `v1.0.0` and the short commit
SHA for manual builds.

## Mount a working directory

Mount a host directory so notebooks and results survive the container:

```bash
docker run --rm -p 8888:8888 \
    -v "$PWD:/home/jovyan/work" \
    ghcr.io/vergaju/evoflow:latest
```

The container starts in `/home/jovyan/work`, so JupyterLab opens the mounted
directory.

## Open the vignette

The pre-installed vignette lives inside the image at:

```text
/home/jovyan/work/vignettes/use_case.ipynb
```

It is the same notebook rendered in {doc}`vignettes/index`. Run it top to
bottom to:

1. infer a signalling network with NicheNet (R),
2. load a single-cell dataset containing NK cells with `pertpy`,
3. run the EvoFlow genetic algorithm on the resulting network,
4. inspect and interpret the knock-out candidates.

The first run downloads the NicheNet reference networks from Zenodo (about
1 GB), so it needs internet access. `pertpy` downloads the `kang_2018` dataset
(about 38 MB) into its local dataset directory.

## Building the image locally

While the repository is still private, build from a checkout:

```bash
docker build -f docker/Dockerfile -t evoflow:dev .
```

The Dockerfile accepts these build arguments:

| Argument           | Default                                             | Meaning                                      |
| ------------------ | --------------------------------------------------- | -------------------------------------------- |
| `BASE_IMAGE`       | `quay.io/jupyter/datascience-notebook:2026-11-02`   | Pinned Jupyter Data Science image            |
| `INSTALL_MODE`     | `local`                                             | `local` builds from the checkout; `pypi` installs `evoflow-bio==EVOFLOW_VERSION` |
| `EVOFLOW_VERSION`  | `1.0.0`                                             | Version installed in `pypi` mode             |
| `NICHENET_REF`     | pinned commit                                       | NicheNet git ref installed from GitHub       |

## What is inside

- **R** and the Jupyter R kernel from the upstream image.
- **NicheNet** (`saeyslab/nichenetr`) and its Bioconductor dependencies.
- **EvoFlow** itself (local or PyPI, depending on `INSTALL_MODE`).
- The vignette notebook and the NicheNet helper script under
  `/home/jovyan/work/vignettes/`.
