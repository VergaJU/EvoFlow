import pickle

import numpy as np
import pandas as pd
import pytest
import yaml

from EvoFlow.model import GAM


class _DummyModel:
    def predict(self, X):
        return np.zeros(X.shape[0])


@pytest.fixture
def minimal_config(tmp_path):
    config = {
        "model_path": str(tmp_path / "model.pkl"),
        "y_col": "score",
        "features": [
            {"name": "log_search_space", "term": "s"},
            {"name": "alpha", "term": "l"},
        ],
        "training_data": str(tmp_path / "training_data.csv"),
        "columns_fix_mapping": [
            {"columns": "alpha", "method": "inplace", "fill_value": 0.0},
        ],
    }
    path = tmp_path / "config.yaml"
    with open(path, "w") as f:
        yaml.dump(config, f)
    return path, config


def test_init_with_config_path(minimal_config):
    path, config = minimal_config
    gam = GAM(config_path=path)
    assert gam.y_col == "score"
    assert gam.features == ["log_search_space", "alpha"]


def test_init_missing_model_path(tmp_path):
    config = {"y_col": "x", "features": [{"name": "f1", "term": "s"}]}
    path = tmp_path / "bad.yaml"
    with open(path, "w") as f:
        yaml.dump(config, f)
    with pytest.raises(ValueError, match="model_path"):
        GAM(config_path=path)


def test_init_missing_y_col(tmp_path):
    config = {"model_path": "m.pkl", "features": [{"name": "f1", "term": "s"}]}
    path = tmp_path / "bad.yaml"
    with open(path, "w") as f:
        yaml.dump(config, f)
    with pytest.raises(ValueError, match="y_col"):
        GAM(config_path=path)


def test_init_missing_features(tmp_path):
    config = {"model_path": "m.pkl", "y_col": "x"}
    path = tmp_path / "bad.yaml"
    with open(path, "w") as f:
        yaml.dump(config, f)
    with pytest.raises(ValueError, match="features"):
        GAM(config_path=path)


def test_init_no_features(tmp_path):
    config = {
        "model_path": "m.pkl",
        "y_col": "x",
        "features": [],
    }
    path = tmp_path / "bad.yaml"
    with open(path, "w") as f:
        yaml.dump(config, f)
    with pytest.raises(ValueError, match="no features"):
        GAM(config_path=path)


def test_init_unsupported_term(tmp_path):
    config = {
        "model_path": "m.pkl",
        "y_col": "x",
        "features": [{"name": "f1", "term": "z"}],
    }
    path = tmp_path / "bad.yaml"
    with open(path, "w") as f:
        yaml.dump(config, f)
    with pytest.raises(ValueError, match="Unsupported"):
        GAM(config_path=path)


def test_load_model_default():
    gam = GAM()
    model = gam.load_model()
    assert model is not None


def test_predict_missing_features(minimal_config):
    path, _ = minimal_config
    gam = GAM(config_path=path)
    gam._model = _DummyModel()
    df = pd.DataFrame({"wrong_col": [1.0]})
    with pytest.raises(ValueError, match="missing"):
        gam.predict(df)


def test_predict_integration():
    gam = GAM()
    gam.load_model()
    df = pd.DataFrame(
        {
            "log_search_space": [2.0],
            "alpha": [1.5],
            "density": [0.05],
            "clustering_coef": [0.1],
            "patience": [200],
            "population": [500],
        }
    )
    scores = gam.predict(df)
    assert isinstance(scores, np.ndarray)
    assert scores.shape == (1,)
    assert 0.0 <= scores[0] <= 1.0


def test_rank_integration(complex_graph):
    from EvoFlow.features_extraction import NetworkFeatures

    nf = NetworkFeatures(G=complex_graph, b=3)
    features = nf.extract_features()
    df = pd.DataFrame(features)

    gam = GAM()
    population, patience = gam.rank(df=df)
    assert isinstance(population, (int, float, np.integer, np.floating))
    assert isinstance(patience, (int, float, np.integer, np.floating))
    assert population > 0
    assert patience > 0


def test_prepare_training_set_transform():
    gam = GAM()
    df = pd.DataFrame(
        {
            "log_search_space": [2.0],
            "alpha": [1.5],
            "density": [0.05],
            "clustering_coef": [0.1],
            "patience": [200],
            "population": [500],
            "score": [0.8],
        }
    )
    X, y, feature_names = gam.prepare_training_set(df)
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert len(feature_names) == 6
    # y = log(1 - 0.8) = log(0.2) ≈ -1.609
    assert abs(y[0] - (-1.6094379)) < 0.001


def test_prepare_training_set_clipping():
    gam = GAM()
    df = pd.DataFrame(
        {
            "log_search_space": [2.0],
            "alpha": [1.5],
            "density": [0.05],
            "clustering_coef": [0.1],
            "patience": [200],
            "population": [500],
            "score": [1.0],
        }
    )
    _, y, _ = gam.prepare_training_set(df)
    # score 1.0 should be clipped to 0.9999 → log(1-0.9999) = log(0.0001) ≈ -9.21
    assert y[0] < -9.0


def test_model_terms_built_correctly(minimal_config):
    path, _ = minimal_config
    gam = GAM(config_path=path)
    model = gam.model()
    assert model is not None
    # 2 features → 2 terms
    assert len(model.terms) == 2


def test_save_model(tmp_path):
    config = {
        "model_path": str(tmp_path / "model.pkl"),
        "y_col": "score",
        "features": [{"name": "x", "term": "l"}],
        "training_data": str(tmp_path / "data.csv"),
        "columns_fix_mapping": [],
    }
    cfg_path = tmp_path / "cfg.yaml"
    with open(cfg_path, "w") as f:
        yaml.dump(config, f)

    gam = GAM(config_path=cfg_path)
    gam._model = _DummyModel()
    scores_df = pd.DataFrame({"RMSE": [0.1], "R2": [0.9]})
    gam.save_model(scores_df)

    saved_path = tmp_path / "model.pkl"
    assert saved_path.exists()
    with open(saved_path, "rb") as f:
        data = pickle.load(f)
    assert "model" in data
    assert "scores" in data
