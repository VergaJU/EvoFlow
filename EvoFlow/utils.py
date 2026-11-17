"""Serialization helpers for EvoFlow result objects.

All helpers write to the **current working directory** unless an absolute path
is passed in ``filename`` and use the file names produced by
:func:`EvoFlow.cli.run_evoflow`.
"""

import json
import pickle

import pandas as pd
import yaml


def save_to_json(data, filename="ga_results.json"):
    """Serialize ``data`` to a JSON file.

    Args:
        data: JSON-serializable object (the GA result dictionary).
        filename: Output path. Defaults to ``"ga_results.json"``.
    """
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)


def save_to_yaml(data, filename="ga_results.yaml"):
    """Serialize ``data`` to a YAML file.

    Args:
        data: YAML-serializable object (the GA result dictionary).
        filename: Output path. Defaults to ``"ga_results.yaml"``.
    """
    with open(filename, "w") as f:
        yaml.dump(data, f, default_flow_style=False)


def save_population(population, filename="ga_population.pkl"):
    """Pickle the final DEAP population.

    Args:
        population: DEAP population (list of individuals).
        filename: Output path. Defaults to ``"ga_population.pkl"``.
    """
    with open(filename, "wb") as f:
        pickle.dump(population, f, protocol=pickle.HIGHEST_PROTOCOL)


def save_logs(logs, filename="ga_logs.pkl"):
    """Pickle the optimization log summary.

    Args:
        logs: Log dictionary returned by :meth:`EvoFlow.core.GAOptimizer.run`.
        filename: Output path. Defaults to ``"ga_logs.pkl"``.
    """
    with open(filename, "wb") as f:
        pickle.dump(logs, f, protocol=pickle.HIGHEST_PROTOCOL)


def save_norm(norm, filename="ga_norm.tsv"):
    """Write the normalized Pareto front to a TSV file.

    Args:
        norm: Array-like of shape ``(n_points, 3)`` with normalized flow,
            number of removed nodes and SCC redundancy.
        filename: Output path. Defaults to ``"ga_norm.tsv"``.
    """
    pd.DataFrame(norm, columns=["Flow", "Nodes", "SCC redundancy"]).to_csv(
        filename, sep="\t"
    )
