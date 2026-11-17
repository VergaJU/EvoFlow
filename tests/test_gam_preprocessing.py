import numpy as np
import pandas as pd
import pytest

from EvoFlow.gam_preprocessing import Preprocessing


@pytest.fixture
def sample_csv(tmp_path):
    data = {
        "log_search_space": [1.0, 2.0, 3.0],
        "alpha": [0.1, 0.2, np.nan],
        "density": [0.5, 0.6, 0.7],
        "clustering_coef": [0.3, 0.4, 0.5],
        "patience": [np.nan, 20, 30],
        "population": [50, 100, 150],
        "budget": [1000, 2000, 3000],
        "score": [0.8, 0.9, 0.85],
    }
    df = pd.DataFrame(data)
    path = tmp_path / "test_data.csv"
    df.to_csv(path, index=False)
    return path, df


def test_load_training_data_not_found():
    pre = Preprocessing(inputh_path="/nonexistent/path.csv")
    with pytest.raises(FileNotFoundError):
        pre.load_training_data()


def test_load_training_data_missing_columns(tmp_path):
    df = pd.DataFrame({"a": [1], "b": [2]})
    path = tmp_path / "bad.csv"
    df.to_csv(path, index=False)
    pre = Preprocessing(
        inputh_path=path,
        columns=["a", "b", "missing"],
    )
    with pytest.raises(ValueError, match="missing"):
        pre.load_training_data()


def test_load_training_data_ok(sample_csv):
    path, _ = sample_csv
    pre = Preprocessing(inputh_path=path)
    df = pre.load_training_data()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty


def test_handle_missing_values_invalid_method(sample_csv):
    path, _ = sample_csv
    pre = Preprocessing(inputh_path=path)
    df = pre.load_training_data()
    with pytest.raises(ValueError, match="Invalid method"):
        pre.handle_missing_values(df, method="invalid")


def test_handle_missing_values_inplace(sample_csv):
    path, _ = sample_csv
    pre = Preprocessing(inputh_path=path)
    df = pre.load_training_data()
    cleaned = pre.handle_missing_values(
        df, method="inplace", subset="alpha", fill_value=0.0
    )
    assert cleaned["alpha"].isna().sum() == 0
    assert cleaned["alpha"].iloc[2] == 0.0


def test_handle_missing_values_na_remove(sample_csv):
    path, _ = sample_csv
    pre = Preprocessing(inputh_path=path)
    df = pre.load_training_data()
    cleaned = pre.handle_missing_values(
        df, method="na_remove", subset=["patience", "budget"]
    )
    assert cleaned["patience"].isna().sum() == 0
    assert len(cleaned) == 2  # Rows with patience=NaN removed


def test_handle_missing_values_inplace_no_fill(sample_csv):
    path, _ = sample_csv
    pre = Preprocessing(inputh_path=path)
    df = pre.load_training_data()
    with pytest.raises(ValueError, match="fill_value"):
        pre.handle_missing_values(df, method="inplace", subset="alpha", fill_value=None)


def test_handle_missing_values_nonexistent_subset(sample_csv):
    path, _ = sample_csv
    pre = Preprocessing(inputh_path=path)
    df = pre.load_training_data()
    with pytest.raises(ValueError, match="subset column"):
        pre.handle_missing_values(
            df, method="inplace", subset="nonexistent", fill_value=0
        )


def test_preprocess_data_pipeline(sample_csv):
    path, _ = sample_csv
    pre = Preprocessing(inputh_path=path)
    columns_fix_mapping = [
        {"columns": "alpha", "method": "inplace", "fill_value": 0.0},
        {"columns": ["patience", "budget"], "method": "na_remove"},
    ]
    cleaned = pre.preprocess_data(columns_fix_mapping=columns_fix_mapping, save=False)
    assert cleaned["alpha"].isna().sum() == 0
    assert cleaned["patience"].isna().sum() == 0


def test_preprocess_data_invalid_step(sample_csv):
    path, _ = sample_csv
    pre = Preprocessing(inputh_path=path)
    with pytest.raises(ValueError, match="columns.*method"):
        pre.preprocess_data(columns_fix_mapping=[{"invalid": True}], save=False)


def test_save_data(tmp_path):
    df = pd.DataFrame({"x": [1, 2]})
    pre = Preprocessing(inputh_path=tmp_path / "ignored.csv")
    out = tmp_path / "saved.csv"
    pre.save_data(df, output_path=out)
    assert out.exists()
    loaded = pd.read_csv(out)
    assert len(loaded) == 2
