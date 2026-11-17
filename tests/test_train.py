import pytest

from EvoFlow.train import load_gam_class


def test_load_gam_class_found(tmp_path):
    model_py = tmp_path / "model.py"
    model_py.write_text("class GAM:\n    pass\n")
    cls = load_gam_class(tmp_path)
    assert cls.__name__ == "GAM"


def test_load_gam_class_found_models(tmp_path):
    """Should also look for models.py."""
    models_py = tmp_path / "models.py"
    models_py.write_text("class GAM:\n    pass\n")
    cls = load_gam_class(tmp_path)
    assert cls.__name__ == "GAM"


def test_load_gam_class_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_gam_class(tmp_path)
