"""Preprocessing pipeline for GAM training data.

:class:`Preprocessing` loads the collected GA tuning dataset, validates that
all expected columns are present and applies declarative missing-value fixes
before the data is handed to :class:`EvoFlow.model.GAM`.
"""

import os
from pathlib import Path
from typing import Any, Literal

import pandas as pd


class Preprocessing:
    """Load, validate and clean the GAM training table.

    Attributes:
        input_path (str | pathlib.Path): Path to the CSV training dataset.
        columns (list[str]): Columns that must be present in the dataset.
    """

    def __init__(
        self,
        inputh_path: str | Path,
        columns: list[str] = [
            "log_search_space",
            "alpha",
            "density",
            "clustering_coef",
            "patience",
            "population",
            "budget",
            "score",
        ],
    ):
        """Configure the preprocessor.

        Args:
            inputh_path: Path to the CSV training dataset (the misspelling is
                kept for backwards compatibility).
            columns: Expected dataset columns. Defaults to the columns used by
                the bundled GAM model.
        """
        self.input_path = inputh_path
        self.columns = columns

    def load_training_data(self, sep=",") -> pd.DataFrame:
        """Read the training CSV and keep the expected columns.

        Args:
            sep: Field separator passed to :func:`pandas.read_csv`.

        Returns:
            pandas.DataFrame: The dataset restricted to ``self.columns``.

        Raises:
            FileNotFoundError: If ``self.input_path`` does not exist.
            ValueError: If any expected column is missing from the file.
        """
        # check path correct:
        if not os.path.exists(self.input_path):
            raise FileNotFoundError(
                f"The file at '{self.input_path}' could not be found!"
            )
        df = pd.read_csv(self.input_path, sep=sep)
        missing_cols = [col for col in self.columns if col not in df.columns]
        if len(missing_cols) > 0:
            raise ValueError(
                f"Dataset validation failed. The following column(s) are missing"
                f"{missing_cols}"
            )
        return df[self.columns]

    def handle_missing_values(
        self,
        df: pd.DataFrame,
        method: Literal["na_remove", "inplace"] = "inplace",
        fill_value: Any | None = 0,
        subset: str | list[str] | None = "alpha",
    ) -> pd.DataFrame:
        """Remove or fill missing values in the selected columns.

        Args:
            df: Dataframe to clean (a copy is returned).
            method: ``"na_remove"`` drops rows with missing values;
                ``"inplace"`` replaces them with ``fill_value``.
            fill_value: Replacement value used by the ``"inplace"`` method.
            subset: Column or columns the operation is restricted to. ``None``
                applies it to the whole dataframe.

        Returns:
            pandas.DataFrame: Cleaned copy of ``df``.

        Raises:
            ValueError: If ``method`` is unknown, if ``subset`` names a missing
                column, or if ``"inplace"`` is used without ``fill_value``.
        """
        allowed_methods = ["na_remove", "inplace"]

        if method not in allowed_methods:
            raise ValueError(
                f"Invalid method '{method}'. Allowed methods are: {allowed_methods}"
            )

        df_clean = df.copy()

        if subset is not None:
            subset_cols = [subset] if isinstance(subset, str) else subset
            missing_cols = [col for col in subset_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(
                    f"The following subset column(s) does not exist in the dataframe:"
                    f"{missing_cols}"
                )

        if method == "na_remove":
            if subset is not None:
                df_clean = df_clean.dropna(subset=subset_cols)
            else:
                df_clean = df_clean.dropna()

        if method == "inplace":
            if fill_value is None:
                raise ValueError(
                    "You must provide a valid 'fill_value' when using 'inplace' method"
                )
            if subset_cols:
                df_clean[subset_cols] = df_clean[subset_cols].fillna(fill_value)
            else:
                df_clean = df_clean.fillna(fill_value)

        return df_clean

    def save_data(self, df, output_path: str | Path) -> None:
        """Write a cleaned dataframe to CSV without the index.

        Args:
            df: Dataframe to write.
            output_path: Destination path.
        """
        df.to_csv(output_path, index=False)

    def preprocess_data(
        self,
        columns_fix_mapping: list[dict[str, Any]],
        output_path: str | Path | None = None,
        save: bool | None = True,
    ) -> pd.DataFrame:
        """Run the full cleaning pipeline.

        Args:
            columns_fix_mapping: Ordered list of steps, each a dict with
                ``columns`` and ``method`` keys plus an optional ``fill_value``
                (see :meth:`handle_missing_values`).
            output_path: Destination CSV used when ``save`` is true.
            save: Whether to write the cleaned dataframe to ``output_path``.

        Returns:
            pandas.DataFrame: The cleaned dataset.

        Raises:
            ValueError: If a step is missing the ``columns`` or ``method`` key.
        """
        df = self.load_training_data()
        for idx, step in enumerate(columns_fix_mapping):
            if "columns" not in step or "method" not in step:
                raise ValueError(
                    f"Validation error at step {idx}: Each step must contain 'columns' and 'method' keys."
                )
            columns = step["columns"]
            method = step["method"]
            fill_value = step.get("fill_value", None)

            df = self.handle_missing_values(
                df=df, method=method, subset=columns, fill_value=fill_value
            )

        if save:
            self.save_data(df=df, output_path=output_path)
        return df


if __name__ == "__main__":
    import tempfile

    import numpy as np

    with tempfile.TemporaryDirectory() as tmp_dir:
        temp_file_path = Path(tmp_dir) / "test_data.csv"
        # Create a tiny test dataframe
        test_data = {
            "log_search_space": [1.0, 2.0, 3.0],
            "alpha": [0.1, 0.2, np.nan],
            "density": [0.5, 0.6, 0.7],
            "clustering_coef": [0.3, 0.4, 0.5],
            "patience": [np.nan, 20, 30],
            "population": [50, 100, 150],
            "budget": [1000, 2000, 3000],
            "score": [0.8, 0.9, 0.85],
        }
        test_df = pd.DataFrame(test_data)
        test_df.to_csv(temp_file_path, index=False)
        columns_fix_mapping_1 = [
            {"columns": "alpha", "method": "inplace", "fill_value": 0.0}
        ]
        columns_fix_mapping_2 = [
            {"columns": ["patience", "budget"], "method": "na_remove"}
        ]
        print("Test DataFrame:")
        print(test_df)
        print("\n" + "=" * 50 + "\n")
        preprocessor = Preprocessing(
            inputh_path=temp_file_path, columns=test_df.columns
        )
        print(
            f"Testing handle_missing_values with 'inplace' method, columns: {columns_fix_mapping_1[0]['columns']}"
        )
        cleaned_df = preprocessor.preprocess_data(
            columns_fix_mapping=columns_fix_mapping_1,
            output_path=Path(tmp_dir) / "processed_test_data.csv",
            save=True,
        )

        print(cleaned_df)
        print("\n" + "=" * 50 + "\n")
        print(
            f"Testing handle_missing_values with 'na_remove' method: {columns_fix_mapping_2[0]['columns']}"
        )

        cleaned_df = preprocessor.preprocess_data(
            columns_fix_mapping=columns_fix_mapping_2,
            output_path=Path(tmp_dir) / "processed_test_data.csv",
            save=True,
        )
        print(cleaned_df)
