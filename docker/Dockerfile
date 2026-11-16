# syntax=docker/dockerfile:1.7
#
# Reproducible R + Python image for the EvoFlow workflow.
#
# Build locally (repository still private):
#   docker build -f docker/Dockerfile -t evoflow:dev .
#
# Build from PyPI instead of the checkout:
#   docker build -f docker/Dockerfile --build-arg INSTALL_MODE=pypi -t evoflow:pypi .

ARG BASE_IMAGE=quay.io/jupyter/datascience-notebook:2026-11-02
FROM ${BASE_IMAGE}

USER root

# System libraries needed by R/Bioconductor packages (curl, git, XML, GSL, GLPK,
# graphics devices, HDF5) and by Python wheels.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libcurl4-openssl-dev \
        libssl-dev \
        libxml2-dev \
        libgit2-dev \
        libglpk-dev \
        libgsl-dev \
        libpng-dev \
        libfreetype6-dev \
        libfontconfig1-dev \
        libharfbuzz-dev \
        libfribidi-dev \
        libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# R: NicheNet (used to infer the signalling network for the vignette)
# ---------------------------------------------------------------------------
ARG NICHENET_REF=66f90d5eeafef280b2b2f339b3fd70ffec1781dd

RUN Rscript -e "install.packages(c('remotes', 'BiocManager', 'optparse'), repos = 'https://cloud.r-project.org')" \
    && Rscript -e "BiocManager::install(ask = FALSE, update = FALSE)" \
    && Rscript -e "remotes::install_github('saeyslab/nichenetr@${NICHENET_REF}', upgrade = 'never', dependencies = TRUE)"

# ---------------------------------------------------------------------------
# Python: EvoFlow (local checkout by default, PyPI optional)
# ---------------------------------------------------------------------------
ARG INSTALL_MODE=local
ARG EVOFLOW_VERSION=1.0.0

COPY pyproject.toml README.md LICENSE CHANGELOG.md /tmp/pkg/
COPY EvoFlow /tmp/pkg/EvoFlow

RUN if [ "${INSTALL_MODE}" = "pypi" ]; then \
        python -m pip install --no-cache-dir "evoflow-bio[vignette]==${EVOFLOW_VERSION}"; \
    else \
        python -m pip install --no-cache-dir "/tmp/pkg[vignette]"; \
    fi \
    && rm -rf /tmp/pkg

# ---------------------------------------------------------------------------
# Vignette assets
# ---------------------------------------------------------------------------
COPY docs/vignettes/ /home/jovyan/work/vignettes/

RUN fix-permissions /home/jovyan/work

USER jovyan
WORKDIR /home/jovyan/work

# Keep Matplotlib's cache and downloaded data inside the mounted workdir.
ENV MPLCONFIGDIR=/home/jovyan/work/.matplotlib
ENV PYTHONUNBUFFERED=1
