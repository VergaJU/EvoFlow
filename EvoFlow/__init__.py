"""EvoFlow: biological network analysis with MaxFlow and genetic algorithms.

EvoFlow identifies critical nodes in gene regulatory networks by combining
maximum-flow / min-cut analysis with a multi-objective genetic algorithm. It
integrates single-cell or bulk expression data with ligand-receptor signalling
networks to score knock-outs that reduce signal flow from ligands to targets.

The public API lives in the submodules:

- :mod:`EvoFlow.cli` -- command-line entry point and :func:`run_evoflow`.
- :mod:`EvoFlow.biology` -- data loading, percolation and network construction.
- :mod:`EvoFlow.preparer` -- SuperSource/SuperSink flow topology.
- :mod:`EvoFlow.core` -- flow computation and the genetic algorithm.
- :mod:`EvoFlow.features_extraction` -- graph features for GAM parameter tuning.
- :mod:`EvoFlow.model` and :mod:`EvoFlow.gam_preprocessing` -- GAM parameter
  estimation.
- :mod:`EvoFlow.visualization` -- plotting and network export helpers.
- :mod:`EvoFlow.utils` -- serialization helpers.
"""

__version__ = "1.0.0"

__all__ = ["__version__"]
