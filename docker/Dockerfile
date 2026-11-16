# syntax=docker/dockerfile:1.7
#
# Reproducible Jupyter image for the EvoFlow workflow.
#
# NicheNet inference is done in Python with NicheNetPy, so no R/Bioconductor
# stack is required.
#
# Build locally (repository still private):
#   docker build -f docker/Dockerfile -t evoflow:dev .
#
# Build from PyPI instead of the checkout:
#   docker build -f docker/Dockerfile --build-arg INSTALL_MODE=pypi -t evoflow:pypi .

ARG BASE_IMAGE=quay.io/jupyter/scipy-notebook:2026-11-02
FROM ${BASE_IMAGE}

USER root

# ---------------------------------------------------------------------------
# Python: EvoFlow + vignette dependencies (NicheNetPy, pertpy)
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
