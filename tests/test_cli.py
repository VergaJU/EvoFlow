import sys

import anndata as ad
import pandas as pd
import pytest
import scipy.sparse as sp

from EvoFlow import cli

BASE_ARGV = [
    "-n",
    "net.tsv",
    "-a",
    "ann.tsv",
    "-f",
    "expr.h5ad",
    "-r",
    "reg.csv",
    "-b",
    "REC1",
    "-m",
    "MED1",
    "-s",
    "ligands.csv",
    "-l",
    "celltype",
    "-c",
    "Type A",
]

EXPECTED_KEYS = {
    "network",
    "annotation",
    "expression_file",
    "regulons",
    "receptors_raw",
    "mediators_raw",
    "ligands_file",
    "label",
    "celltype",
    "lock_targets",
    "expression_stat",
    "expression_norm",
    "expression_transpose",
    "generations",
    "population_size",
    "early_stop_gen",
    "guess_parameters",
    "delta_penalty",
    "repair",
    "norm_hypervolume",
    "dynamic",
    "n_jobs",
}


def _capture_main(monkeypatch, extra_args=(), base_argv=None):
    captured = {}

    def fake_run_evoflow(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(cli, "run_evoflow", fake_run_evoflow)
    monkeypatch.setattr(
        sys, "argv", ["evoflow", *(base_argv or BASE_ARGV), *extra_args]
    )
    cli.main()
    return captured


def test_main_forwards_defaults(monkeypatch):
    captured = _capture_main(monkeypatch)
    assert set(captured) == EXPECTED_KEYS
    assert captured["expression_file"] == "expr.h5ad"
    assert captured["population_size"] is None
    assert captured["early_stop_gen"] is None
    assert captured["guess_parameters"] is True
    assert captured["delta_penalty"] is False
    assert captured["repair"] is True
    assert captured["norm_hypervolume"] is False
    assert captured["dynamic"] is True
    assert captured["n_jobs"] is None
    assert captured["lock_targets"] == 0
    assert captured["generations"] == 10000
    assert captured["receptors_raw"] == "REC1"
    assert captured["mediators_raw"] == "MED1"
    assert captured["network"] == "net.tsv"
    assert captured["annotation"] == "ann.tsv"
    assert captured["regulons"] == "reg.csv"
    assert captured["ligands_file"] == "ligands.csv"
    assert captured["label"] == "celltype"
    assert captured["celltype"] == "Type A"
    assert captured["expression_stat"] == "nonzero_ratio"
    assert captured["expression_norm"] == "none"
    assert captured["expression_transpose"] is False


def test_main_forwards_values_and_boolean_negation(monkeypatch):
    captured = _capture_main(
        monkeypatch,
        [
            "--pop_size",
            "5",
            "--early_stop",
            "20",
            "--generations",
            "3",
            "--n_jobs",
            "2",
            "--lock_targets",
            "3",
            "--no-repair",
            "--delta_penalty",
            "--no-dynamic_rates",
            "--norm_hypervolume",
            "--expression_stat",
            "mean",
            "--expression_norm",
            "cpm",
            "--expression_transpose",
        ],
    )
    assert set(captured) == EXPECTED_KEYS
    assert captured["population_size"] == 5
    assert captured["early_stop_gen"] == 20
    assert captured["generations"] == 3
    assert captured["n_jobs"] == 2
    assert captured["lock_targets"] == 3
    assert captured["repair"] is False
    assert captured["delta_penalty"] is True
    assert captured["dynamic"] is False
    assert captured["norm_hypervolume"] is True
    assert captured["expression_stat"] == "mean"
    assert captured["expression_norm"] == "cpm"
    assert captured["expression_transpose"] is True


def _assert_main_exits(monkeypatch, extra_args, base_argv=None):
    called = []
    monkeypatch.setattr(cli, "run_evoflow", lambda **kw: called.append(kw))
    monkeypatch.setattr(
        sys, "argv", ["evoflow", *(base_argv or BASE_ARGV), *extra_args]
    )
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2
    assert called == []


def test_main_rejects_pop_size_without_early_stop(monkeypatch):
    _assert_main_exits(monkeypatch, ["--pop_size", "5"])


def test_main_rejects_early_stop_without_pop_size(monkeypatch):
    _assert_main_exits(monkeypatch, ["--early_stop", "20"])


def test_main_rejects_no_guess_without_values(monkeypatch):
    _assert_main_exits(monkeypatch, ["--no-guess_parameters"])


def test_main_rejects_invalid_expression_stat(monkeypatch):
    _assert_main_exits(monkeypatch, ["--expression_stat", "mode"])


def test_main_rejects_invalid_expression_norm(monkeypatch):
    _assert_main_exits(monkeypatch, ["--expression_norm", "quantile"])


def _base_argv_without_label():
    return BASE_ARGV[: BASE_ARGV.index("-l")]


def test_main_rejects_celltype_without_label(monkeypatch):
    _assert_main_exits(
        monkeypatch, ["-c", "Type A"], base_argv=_base_argv_without_label()
    )


def test_main_rejects_label_without_celltype(monkeypatch):
    # BASE_ARGV minus the trailing "-c" "Type A" pair
    _assert_main_exits(monkeypatch, [], base_argv=BASE_ARGV[: BASE_ARGV.index("-c")])


def test_main_allows_missing_label_and_celltype(monkeypatch):
    captured = _capture_main(monkeypatch, base_argv=_base_argv_without_label())
    assert set(captured) == EXPECTED_KEYS
    assert captured["label"] is None
    assert captured["celltype"] is None


def test_run_evoflow_rejects_invalid_expression_stat():
    with pytest.raises(ValueError, match="expression_stat"):
        cli.run_evoflow(
            network="n",
            annotation="a",
            expression_file="e.h5ad",
            expression_stat="mode",
        )


def test_run_evoflow_rejects_invalid_expression_norm():
    with pytest.raises(ValueError, match="expression_norm"):
        cli.run_evoflow(
            network="n",
            annotation="a",
            expression_file="e.h5ad",
            expression_norm="quantile",
        )


def test_run_evoflow_rejects_half_label_pair():
    with pytest.raises(ValueError, match="together"):
        cli.run_evoflow(
            network="n",
            annotation="a",
            expression_file="e.h5ad",
            label="celltype",
        )
    with pytest.raises(ValueError, match="together"):
        cli.run_evoflow(
            network="n",
            annotation="a",
            expression_file="e.h5ad",
            celltype="Type A",
        )


def test_run_evoflow_rejects_half_specified_pair():
    with pytest.raises(ValueError, match="together"):
        cli.run_evoflow(
            network="n", annotation="a", expression_file="e.h5ad", population_size=5
        )
    with pytest.raises(ValueError, match="together"):
        cli.run_evoflow(
            network="n", annotation="a", expression_file="e.h5ad", early_stop_gen=10
        )


def test_run_evoflow_end_to_end(tmp_path, monkeypatch):
    (tmp_path / "net.tsv").write_text(
        "start\tto\tweight\tlayer\n"
        "L1\tR1\t1.0\tsignaling\n"
        "R1\tM1\t1.0\tsignaling\n"
        "M1\tT1\t1.0\tsignaling\n"
    )
    (tmp_path / "ann.tsv").write_text(
        "gene\tnode_type\nL1\tligand\nR1\treceptor\nM1\tmediator\nT1\ttarget\n"
    )
    (tmp_path / "ligands.csv").write_text("x\nL1\n")
    adata = ad.AnnData(
        X=sp.csr_matrix([[1, 1, 1, 1], [1, 0, 1, 0]]),
        var=pd.DataFrame(index=["L1", "R1", "M1", "T1"]),
        obs=pd.DataFrame({"celltype": ["Type A", "Type A"]}),
    )
    adata.write_h5ad(tmp_path / "expr.h5ad")

    monkeypatch.chdir(tmp_path)

    # guess_parameters stays at its True default while population_size/early_stop_gen
    # are provided: run_evoflow must auto-disable guessing (otherwise run() raises).
    prefix = cli.run_evoflow(
        network=str(tmp_path / "net.tsv"),
        annotation=str(tmp_path / "ann.tsv"),
        expression_file=str(tmp_path / "expr.h5ad"),
        ligands_file=str(tmp_path / "ligands.csv"),
        targets="T1",
        label="celltype",
        celltype="Type A",
        generations=2,
        population_size=6,
        early_stop_gen=10,
        use_multiprocessing=False,
        n_jobs=1,
    )

    assert prefix == "expr"
    assert (tmp_path / "expr_ga_results.json").is_file()
    assert (tmp_path / "expr_ga_results.yaml").is_file()
    # save_network/plot functions concatenate the prefix directly (no
    # separator); real prefixes end with "_" after the "tmp" truncation.
    assert (tmp_path / "exprnetwork.graphml").is_file()
    assert (tmp_path / "exprnetwork.cyjs").is_file()
    assert list(tmp_path.glob("ga_run_*.jsonl"))


def test_run_evoflow_bulk_csv_e2e(tmp_path, monkeypatch):
    """End-to-end run on a genes-as-rows bulk CSV (no AnnData, no label/celltype)."""
    (tmp_path / "net.tsv").write_text(
        "start\tto\tweight\tlayer\n"
        "L1\tR1\t1.0\tsignaling\n"
        "R1\tM1\t1.0\tsignaling\n"
        "M1\tT1\t1.0\tsignaling\n"
    )
    (tmp_path / "ann.tsv").write_text(
        "gene\tnode_type\nL1\tligand\nR1\treceptor\nM1\tmediator\nT1\ttarget\n"
    )
    (tmp_path / "ligands.csv").write_text("x\nL1\n")
    (tmp_path / "bulkexpr.csv").write_text(
        "gene,S1,S2,S3\nL1,10,0,4\nR1,5,5,0\nM1,0,3,7\nT1,1,1,1\n"
    )

    monkeypatch.chdir(tmp_path)

    prefix = cli.run_evoflow(
        network=str(tmp_path / "net.tsv"),
        annotation=str(tmp_path / "ann.tsv"),
        expression_file=str(tmp_path / "bulkexpr.csv"),
        ligands_file=str(tmp_path / "ligands.csv"),
        targets="T1",
        expression_transpose=True,
        expression_norm="cpm",
        expression_stat="mean",
        generations=2,
        population_size=6,
        early_stop_gen=10,
        use_multiprocessing=False,
        n_jobs=1,
    )

    assert prefix == "bulkexpr"
    assert (tmp_path / "bulkexpr_ga_results.json").is_file()
    assert (tmp_path / "bulkexpr_ga_results.yaml").is_file()
    assert (tmp_path / "bulkexprnetwork.graphml").is_file()
    assert list(tmp_path.glob("ga_run_*.jsonl"))
