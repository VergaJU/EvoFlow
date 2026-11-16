# API reference

This reference is generated automatically from the package docstrings. Every
public module, class and function is documented below; use the search box or
the sidebar to jump to a specific symbol.

## Module summary

```{eval-rst}
.. autosummary::
   :nosignatures:

   EvoFlow
   EvoFlow.cli
   EvoFlow.biology
   EvoFlow.preparer
   EvoFlow.core
   EvoFlow.features_extraction
   EvoFlow.model
   EvoFlow.gam_preprocessing
   EvoFlow.visualization
   EvoFlow.utils
   EvoFlow.train
```

## Modules

```{eval-rst}
.. automodule:: EvoFlow
   :members:
   :show-inheritance:

.. automodule:: EvoFlow.cli
   :members:
   :show-inheritance:

.. automodule:: EvoFlow.biology
   :members:
   :show-inheritance:

.. automodule:: EvoFlow.preparer
   :members:
   :show-inheritance:

.. automodule:: EvoFlow.core
   :members:
   :show-inheritance:

.. automodule:: EvoFlow.features_extraction
   :members:
   :show-inheritance:

.. automodule:: EvoFlow.model
   :members:
   :show-inheritance:

.. automodule:: EvoFlow.gam_preprocessing
   :members:
   :show-inheritance:

.. automodule:: EvoFlow.visualization
   :members:
   :show-inheritance:

.. automodule:: EvoFlow.utils
   :members:
   :show-inheritance:

.. automodule:: EvoFlow.train
   :members:
   :show-inheritance:
```

## Typical low-level workflow

The example below mirrors the use-case vignette and the reference Snakemake
pipeline:

```python
from EvoFlow.biology import BioNetworkPreparer
from EvoFlow.core import GAOptimizer

preparer = BioNetworkPreparer(lock_targets=1)
preparer.load_data(network_path, annotation_path, expression_path=expression)
preparer.compute_percolation(stat="nonzero_ratio")
preparer.build_base_network()
preparer.update_edge_capacities(method="multiplicative")
sources, sinks = preparer.prepare_bioflow_topology(
    receptors=receptors, target_genes=targets
)

optimizer = GAOptimizer(preparer.G, sources, sinks, max_remove=10, repair=True)
results, logs, population, norm = optimizer.run(
    generations=2000, population_size=100, early_stop_gen=50
)
```
