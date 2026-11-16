"""GAM utilities for training, saving and using a pygam LinearGAM.

This module provides the `GAM` class which wraps common operations:
- loading and preprocessing training data,
- constructing a LinearGAM with chosen term constraints,
- training with cross-validation and grid search,
- saving/loading the trained model, and
- predicting and ranking parameter combinations.

The implementations assume the target variable is a score in (0,1)
and apply the transform `y -> log(1-y)` before training.
"""

from pygam.pygam import LinearGAM, s, l, f
import pandas as pd
import numpy as np
from typing import Any, List, Dict, Tuple
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
import pickle
from EvoFlow.gam_preprocessing import Preprocessing
import itertools
import yaml
from importlib import resources


_TERM_BUILDERS = {
    "s": s,
    "l": l,
    "f": f,
}


class GAM:
    """Thin wrapper around a :class:`pygam.LinearGAM`.

    Handles config loading (from the packaged ``EvoFlow/models/config.yaml`` by
    default), training-data preprocessing, cross-validated training,
    persistence and prediction/ranking of GA hyper-parameter combinations.

    Attributes:
        features (list[str]): Feature names in the order expected by the model.
        y_col (str): Name of the target column.
        data_path (pathlib.Path | None): Training data path from the config.
        model_path (pathlib.Path | None): Model path for user-supplied configs;
            ``None`` when the packaged model is used.
    """

    def __init__(self, config_path: str | Path = None):
        """Create a `GAM` wrapper from a YAML config file.

        The config defines `model_path`, `y_col`, and `features` (an
        ordered list of dicts, one per feature, each with at least a
        `name` and `term` key, and optionally `constraints` and `lam`).
        Term order in the config determines term order in the model,
        so it must match the column order used at training/prediction
        time.

        Args:
            config_path: Path to the YAML config file.

        Raises:
            ValueError: If the config is missing required keys or a
                feature entry uses an unsupported term type.
        """
        self._model = None
        self._default_model_resource = None

        if config_path is None:
            config_text = (
                resources.files("EvoFlow.models").joinpath("config.yaml").read_text()
            )
            config = yaml.safe_load(config_text)
            self._default_model_resource = resources.files("EvoFlow.models").joinpath(
                "model.pkl"
            )
            self.config_path = None
        else:
            self.config_path = Path(config_path)
            with open(self.config_path, "r") as f:
                config = yaml.safe_load(f)

        for required_key in ("model_path", "y_col", "features"):
            if required_key not in config:
                raise ValueError(
                    f"GAM config '{self.config_path}' is missing required "
                    f"key '{required_key}'"
                )

        if self._default_model_resource is None:
            raw_model_path = Path(config["model_path"])
            if not raw_model_path.is_absolute() and self.config_path is not None:
                self.model_path = self.config_path.parent / raw_model_path
            else:
                self.model_path = raw_model_path
        else:
            self.model_path = None

        self.y_col: str = config["y_col"]
        self._feature_config: List[Dict[str, Any]] = config["features"]

        if len(self._feature_config) == 0:
            raise ValueError(f"GAM config '{self.config_path}' defines no features")

        for entry in self._feature_config:
            if "name" not in entry or "term" not in entry:
                raise ValueError(
                    "Each feature entry in the config must define at least "
                    f"'name' and 'term'. Got: {entry}"
                )
            if entry["term"] not in _TERM_BUILDERS:
                raise ValueError(
                    f"Unsupported term type '{entry['term']}' for feature "
                    f"'{entry['name']}'. Supported: {list(_TERM_BUILDERS)}"
                )

        # Ordered list of feature names, derived from the config so the
        # two can never drift apart.
        self.features: List[str] = [entry["name"] for entry in self._feature_config]

        if "training_data" in config:
            raw_data_path = Path(config["training_data"])
            if not raw_data_path.is_absolute() and self.config_path is not None:
                self.data_path = self.config_path.parent / raw_data_path
            else:
                self.data_path = raw_data_path
        else:
            self.data_path = None

        self._columns_mapping: List[Dict[str, Any]] = config["columns_fix_mapping"]

        self.columns_fix_mapping: List[Dict] = []
        for entry in self._columns_mapping:
            if "columns" not in entry or "method" not in entry:
                raise ValueError(
                    f"Column fix mapping require at elast one column and a method"
                    f"got {entry}"
                )
            if entry["method"] == "inplace" and "fill_value" not in entry:
                raise ValueError(
                    f"If method is inplace, please specify which value to inplace"
                    f"got {entry}"
                )
            if entry["method"] == "inplace":
                column_fix = {
                    "columns": entry["columns"],
                    "method": entry["method"],
                    "fill_value": entry["fill_value"],
                }
            else:
                column_fix = {"columns": entry["columns"], "method": entry["method"]}
            self.columns_fix_mapping.append(column_fix)

    def load_model(self):
        """Load a pickled model from `self.model_path` and return it.

        Returns:
            The unpickled model object.

        Raises:
            FileNotFoundError: If the model file does not exist.
            pickle.UnpicklingError: If the model file is corrupted.
        """
        # Scenario A: Loading default packaged model
        if self._default_model_resource is not None:
            try:
                with self._default_model_resource.open("rb") as f:
                    loaded = pickle.load(f)
            except Exception as e:
                raise RuntimeError(f"Failed to load default package model: {e}")

            if isinstance(loaded, dict) and "model" in loaded:
                self._model = loaded["model"]
            else:
                self._model = loaded
            return self._model

        # Scenario B: Loading a user-specified model
        try:
            with open(self.model_path, "rb") as f:
                loaded = pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Model file not found at '{self.model_path}'. "
                "Please ensure the model has been trained and saved."
            )
        except pickle.UnpicklingError as e:
            raise pickle.UnpicklingError(
                f"Failed to load model from '{self.model_path}': the file may be corrupted. "
                f"Error: {e}"
            )
        except Exception as e:
            raise RuntimeError(
                f"Unexpected error loading model from '{self.model_path}': {e}"
            )

        if isinstance(loaded, dict) and "model" in loaded:
            self._model = loaded["model"]
        else:
            self._model = loaded
        return self._model

    def load_data(self):
        """Load and preprocess data from ``self.data_path``.

        Uses :class:`EvoFlow.gam_preprocessing.Preprocessing` to apply the
        configured column fixes and return a cleaned dataframe containing the
        feature columns and the target column.

        Returns:
            pandas.DataFrame: Data ready for training.

        Raises:
            RuntimeError: If the config does not define a training data path.
        """
        if self.data_path is None:
            raise RuntimeError(
                "GAM config does not define 'training_data'; cannot load data."
            )
        preprocesser = Preprocessing(
            inputh_path=self.data_path, columns=self.features + [self.y_col]
        )
        df = preprocesser.preprocess_data(columns_fix_mapping=self.columns_fix_mapping)
        return df

    def prepare_training_set(self, df: pd.DataFrame):
        """Prepare feature matrix `X`, transformed target `y` and feature names.

        The method applies the transformation `y = log(1 - y)` which
        is the representation used for GAM training in this project.

        Args:
            df: DataFrame containing features and the target column.

        Returns:
            Tuple of `(X, y, feature_names)` where `X` is a numpy array,
            `y` is the transformed target array, and `feature_names` is
            a list of column names used as features.
        """
        y = df[self.y_col].values
        # clip y values 0-0.9999 to avoid log(0)
        y = np.clip(y, a_min=0, a_max=0.9999)
        # transform y values as Log(Loss) function
        y = np.log(1 - y)
        X = df.drop(self.y_col, axis=1).to_numpy()
        feature_names = [col for col in df.columns if col != self.y_col]
        return X, y, feature_names

    def model(self, lams: List = None):
        """Build and return a configured `LinearGAM` instance.

        Term types, constraints, and (optionally) per-term `lam` are
        read from the config supplied at construction time, so the
        architecture lives in `gam_config.yaml` rather than being
        hardcoded here. Term order matches `self.features` order.

        Args:
            lams: Optional list of lambda values, one per feature, in
                the same order as `self.features`. Overrides any `lam`
                set in the config for that term.

        Returns:
            An untrained `LinearGAM` instance built from the config.
        """
        terms = None
        for idx, entry in enumerate(self._feature_config):
            term_builder = _TERM_BUILDERS[entry["term"]]

            kwargs: Dict[str, Any] = {}
            if lams is not None:
                kwargs["lam"] = lams[idx]
            elif "lam" in entry and entry["lam"] is not None:
                kwargs["lam"] = entry["lam"]

            if "constraints" in entry and entry["constraints"] is not None:
                kwargs["constraints"] = entry["constraints"]

            term = term_builder(idx, **kwargs)
            terms = term if terms is None else terms + term

        return LinearGAM(terms)

    def train(self, X: np.array, y: np.ndarray):
        """Train the GAM using 10-fold cross-validation then on full data.

        Performs grid search on each CV training fold (GCV internally)
        and collects RMSE and R2 metrics. CV scores are written to
        `cv_scores.csv`. Finally the model is trained on the full
        dataset and returned alongside the CV scores DataFrame.

        Args:
            X: Numpy array of features.
            y: Numpy array of targets (transformed).

        Returns:
            A tuple `(gam, scores)` with the trained `LinearGAM` and a
            pandas DataFrame of CV scores.
        """
        # scores path
        scores_path = self.model_path
        if scores_path is None and self._default_model_resource is not None:
            scores_path = self._default_model_resource
        scores_path = scores_path.with_name(f"{scores_path.stem}_scores.csv")
        # Set up 10-Fold Cross-Validation
        kf = KFold(n_splits=10, shuffle=True, random_state=42)

        rmse_scores = []
        r2_scores = []

        print("Starting 10-Fold Cross-Validation...")
        for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            gam = self.model()

            # Perform Grid Search on the training fold (internally optimized via GCV)
            gam.gridsearch(X_train, y_train, progress=False)

            # Predict on the unseen test fold
            predictions = gam.predict(X_test)
            # Evaluate performance
            fold_rmse = np.sqrt(mean_squared_error(y_test, predictions))
            fold_r2 = r2_score(y_test, predictions)

            rmse_scores.append(fold_rmse)
            r2_scores.append(fold_r2)

            print(f"Fold {fold + 1} - RMSE: {fold_rmse:.4f}, R2: {fold_r2:.4f}")

        print("\n--- Final CV Results ---")
        print(f"Mean RMSE: {np.mean(rmse_scores):.4f} (+/- {np.std(rmse_scores):.4f})")
        print(f"Mean R2:   {np.mean(r2_scores):.4f} (+/- {np.std(r2_scores):.4f})")
        scores = pd.DataFrame({"RMSE": rmse_scores, "R2": r2_scores})
        scores.to_csv(scores_path, index=False)

        ## Train with full data:
        gam = self.model()
        gam.gridsearch(X, y, progress=True)
        self._model = gam
        return gam, scores

    def save_model(self, scores: pd.DataFrame):
        """Save the trained model and associated CV scores to disk.

        The model stored in ``self._model`` and ``scores`` are pickled together
        as ``{"model": ..., "scores": ...}``.

        Args:
            scores (pandas.DataFrame): CV scores to persist alongside the model.
        """
        model_data = {"model": self._model, "scores": scores}

        save_path = self.model_path
        if save_path is None and self._default_model_resource is not None:
            save_path = self._default_model_resource

        with open(save_path, "wb") as f:
            pickle.dump(model_data, f)

    def predict(self, df: pd.DataFrame):
        """Predict scores for rows in ``df`` with the loaded model.

        Validates that ``df`` contains the expected feature columns, computes
        predictions in transformed space and converts them back to the original
        score space.

        Args:
            df (pandas.DataFrame): Dataframe containing the feature columns.

        Returns:
            numpy.ndarray: Predicted scores in the original (untransformed)
            scale.

        Raises:
            ValueError: If any expected feature column is missing.
        """

        if self._model is None:
            self.load_model()

        # verify correct features:
        missing_cols = [col for col in self.features if col not in df.columns]
        if len(missing_cols) > 0:
            raise ValueError(
                f"Dataset validation failed. The following column(s) are missing"
                f"{missing_cols}"
            )
        # ensure correct order:
        df = df[self.features]

        X_test = df.to_numpy()

        predicted = self._model.predict(X_test)

        # recovert predicted to score:
        predicted = 1 - np.exp(predicted)
        return predicted

    def rank(
        self,
        df: pd.DataFrame,
        patience_range: Tuple[int] = (100, 500),
        population_range: Tuple[int] = (50, 2000),
    ):
        """Search parameter grid for best `(population, patience)` pair.

        Constructs a grid of `patience` and `population` values, cross-
        joins it with `df` to form complete candidate rows, predicts
        scores for each candidate and returns the most promising
        `(population, patience)` pair (with a small tolerance on max).

        Args:
            df: Base DataFrame containing other required feature columns.
            patience_range: Inclusive (min, max) range for patience values.
            population_range: Inclusive (min, max) range for population values.

        Returns:
            Tuple `(population, patience)` recommending integer settings.
        """
        patience_options = np.linspace(patience_range[0], patience_range[1], 10)
        population_options = np.linspace(population_range[0], population_range[1], 30)

        comb_df = pd.DataFrame(
            list(itertools.product(patience_options, population_options)),
            columns=["patience", "population"],
        )

        query_df = comb_df.merge(df, how="cross")

        predicted = self.predict(df=query_df)
        query_df["predicted_score"] = predicted
        max_score = query_df["predicted_score"].max()
        threshold_score = max_score - 0.01  # 1% tolerance
        optimal_setups = query_df[query_df["predicted_score"] >= threshold_score]
        best_recommendation = optimal_setups.sort_values(
            by=["population", "patience"]
        ).iloc[0]
        return best_recommendation["population"], best_recommendation["patience"]
