"""Train and save a GAM model from the workspace training data."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Type


def load_gam_class(module_path: Path) -> Type:
    """Load the `GAM` class from either `model.py` or `models.py`."""
    candidates = [module_path / "model.py", module_path / "models.py"]
    for candidate in candidates:
        if candidate.exists():
            spec = importlib.util.spec_from_file_location("gam_module", candidate)
            if spec is None or spec.loader is None:
                raise ImportError(f"Could not load module spec for {candidate}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            return getattr(module, "GAM")

    raise FileNotFoundError(
        "Could not find a GAM module at src/model.py or src/models.py"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and save a GAM model")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "models/config.yaml",
        help="Path to the configuartion file",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    src_dir = project_root
    sys.path.insert(0, str(src_dir))

    GAM = load_gam_class(src_dir)
    model = GAM(config_path=args.config)

    print(f"Loading training data from {model.data_path}")
    df = model.load_data()

    print("Preparing training set")
    X, y, feature_names = model.prepare_training_set(df)
    print(f"Using features: {feature_names}")

    print("Starting training")
    gam, scores = model.train(X, y)

    print(f"Saving model to {model.model_path}")
    model.save_model(scores)
    print("Training complete")


if __name__ == "__main__":
    main()
