"""
Tests for the new prepare_network() orchestration method in BioNetworkPreparer.
"""

import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import networkx as nx
from EvoFlow.biology import BioNetworkPreparer


class TestPrepareNetworkValidation:
    """Tests for input validation in prepare_network()"""

    def test_empty_ligands_raises_error(self, mock_net_df, mock_annotation):
        """Test that empty ligands list raises ValueError"""
        prep = BioNetworkPreparer()

        with pytest.raises(ValueError, match="ligands list cannot be empty"):
            prep.prepare_network(
                network_path="network.txt",
                annotation_path="annotation.txt",
                expression_path="expression.h5ad",
                ligands=[],
                mediators_list=["B"],
                regulons_path="regulons.csv",
            )

    def test_empty_mediators_raises_error(self, mock_net_df, mock_annotation):
        """Test that empty mediators_list raises ValueError"""
        prep = BioNetworkPreparer()

        with pytest.raises(ValueError, match="mediators_list cannot be empty"):
            prep.prepare_network(
                network_path="network.txt",
                annotation_path="annotation.txt",
                expression_path="expression.h5ad",
                ligands=["A"],
                mediators_list=[],
                regulons_path="regulons.csv",
            )

    def test_no_targets_or_regulons_raises_error(self, mock_net_df, mock_annotation):
        """Test that missing both targets and regulons raises ValueError"""
        prep = BioNetworkPreparer()

        with pytest.raises(
            ValueError, match="Either regulons_path or targets must be provided"
        ):
            prep.prepare_network(
                network_path="network.txt",
                annotation_path="annotation.txt",
                expression_path="expression.h5ad",
                ligands=["A"],
                mediators_list=["B"],
                regulons_path=None,
                targets=None,
            )


class TestPrepareNetworkWorkflow:
    """Tests for prepare_network() orchestration workflow"""

    def test_prepare_network_with_regulons_returns_tuple(
        self, mock_net_df, mock_annotation, mock_anndata
    ):
        """Test that prepare_network returns (graph, sources, sinks) tuple"""
        prep = BioNetworkPreparer()

        with patch("pandas.read_csv") as mock_read_csv:

            def read_csv_side_effect(path, **kwargs):
                if "network" in path:
                    return mock_net_df
                elif "annotation" in path:
                    return mock_annotation
                elif "regulons" in path:
                    return pd.DataFrame(
                        {"mediator_col": ["B", "C"], "target_col": ["D", "E"]}
                    )
                return pd.DataFrame()

            mock_read_csv.side_effect = read_csv_side_effect

            with patch("scanpy.read") as mock_sc_read:
                mock_sc_read.return_value = mock_anndata

                with patch.object(prep, "prepare_bioflow_topology") as mock_topo:
                    mock_topo.return_value = (["B"], ["E"])

                    result = prep.prepare_network(
                        network_path="network.txt",
                        annotation_path="annotation.txt",
                        expression_path="file.h5ad",
                        ligands=["A"],
                        mediators_list=["B", "C"],
                        label="cell_type_col",
                        celltype="Type X",
                        regulons_path="regulons.csv",
                        targets=None,
                    )

                    assert isinstance(result, tuple)
                    assert len(result) == 3
                    graph, sources, sinks = result
                    assert isinstance(graph, nx.DiGraph)
                    assert sources == ["B"]
                    assert sinks == ["E"]

    def test_prepare_network_with_targets_returns_tuple(
        self, mock_net_df, mock_annotation, mock_anndata
    ):
        """Test that prepare_network works with explicit targets instead of regulons"""
        prep = BioNetworkPreparer()

        with patch("pandas.read_csv") as mock_read_csv:

            def read_csv_side_effect(path, **kwargs):
                if "network" in path:
                    return mock_net_df
                elif "annotation" in path:
                    return mock_annotation
                return pd.DataFrame()

            mock_read_csv.side_effect = read_csv_side_effect

            with patch("scanpy.read") as mock_sc_read:
                mock_sc_read.return_value = mock_anndata

                with patch.object(prep, "prepare_bioflow_topology") as mock_topo:
                    mock_topo.return_value = (["B"], ["E", "D"])

                    result = prep.prepare_network(
                        network_path="network.txt",
                        annotation_path="annotation.txt",
                        expression_path="file.h5ad",
                        ligands=["A"],
                        mediators_list=["B"],
                        label="cell_type_col",
                        celltype="Type X",
                        regulons_path=None,
                        targets=["D", "E"],
                    )

                    assert isinstance(result, tuple)
                    assert len(result) == 3
                    graph, sources, sinks = result
                    assert isinstance(graph, nx.DiGraph)

    def test_prepare_network_calls_methods_in_order(
        self, mock_net_df, mock_annotation, mock_anndata
    ):
        """Test that prepare_network calls methods in correct sequence"""
        prep = BioNetworkPreparer()

        with patch.object(prep, "load_data") as mock_load:
            with patch.object(prep, "filter_expression_data") as mock_filter:
                with patch.object(prep, "compute_percolation") as mock_perc:
                    with patch.object(prep, "build_base_network") as mock_build:
                        with patch.object(prep, "update_edge_capacities") as mock_cap:
                            with patch.object(prep, "identify_targets") as mock_id:
                                with patch.object(
                                    prep, "prepare_bioflow_topology"
                                ) as mock_topo:
                                    # Set expression to non-AnnData to skip filter_expression_data path
                                    prep.expression = pd.DataFrame()
                                    mock_id.return_value = ["E"]
                                    mock_topo.return_value = (["B"], ["E"])

                                    prep.prepare_network(
                                        network_path="network.txt",
                                        annotation_path="annotation.txt",
                                        expression_path="file.h5ad",
                                        ligands=["A"],
                                        mediators_list=["B"],
                                        label="cell_type_col",
                                        celltype="Type X",
                                        regulons_path="regulons.csv",
                                        targets=None,
                                    )

                                    # Verify methods were called
                                    mock_load.assert_called_once()
                                    # filter_expression_data is skipped because expression is not AnnData
                                    assert mock_filter.call_count == 0
                                    mock_perc.assert_called_once()
                                    mock_build.assert_called_once()
                                    mock_cap.assert_called_once()
                                    mock_id.assert_called_once()
                                    mock_topo.assert_called_once()

    def test_prepare_network_warns_when_label_given_without_anndata(
        self, mock_net_df, mock_annotation, capsys
    ):
        """label/celltype with non-AnnData expression is ignored with a notice"""
        prep = BioNetworkPreparer()

        with (
            patch.object(prep, "load_data"),
            patch.object(prep, "filter_expression_data") as mock_filter,
            patch.object(prep, "compute_percolation"),
            patch.object(prep, "build_base_network"),
            patch.object(prep, "update_edge_capacities"),
            patch.object(prep, "prepare_bioflow_topology") as mock_topo,
        ):
            # Bulk expression: DataFrame, not AnnData
            prep.expression = pd.DataFrame()
            mock_topo.return_value = (["B"], ["E"])

            prep.prepare_network(
                network_path="network.txt",
                annotation_path="annotation.txt",
                expression_path="expression.csv",
                ligands=["A"],
                targets=["D", "E"],
                label="celltype",
                celltype="Type A",
            )

            assert mock_filter.call_count == 0
            assert "only applies to h5ad" in capsys.readouterr().out

    def test_prepare_network_backward_compatibility_individual_methods(
        self, mock_net_df, mock_annotation
    ):
        """Test that individual methods still work for backward compatibility"""
        prep = BioNetworkPreparer()

        with patch("pandas.read_csv") as mock_read_csv:

            def read_csv_side_effect(path, **kwargs):
                if "network" in path:
                    return mock_net_df
                elif "annotation" in path:
                    return mock_annotation
                return pd.DataFrame()

            mock_read_csv.side_effect = read_csv_side_effect

            with patch("scanpy.read") as mock_sc_read:
                mock_sc_read.return_value = MagicMock()

                # Call load_data individually (backward compatibility test)
                prep.load_data("network.txt", "annotation.txt", "file.h5ad", None)

                assert prep.net_df is not None
                assert prep.annotation is not None
