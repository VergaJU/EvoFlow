# Vignettes

End-to-end examples that run the full NicheNet + EvoFlow workflow. They are
designed to be executed inside the provided container
({doc}`../container`), where EvoFlow, NicheNetPy and pertpy are pre-installed.

```{toctree}
:maxdepth: 1

use_case
```

## Use case: IFN-β-stimulated NK cells

{doc}`use_case` answers two related questions with the same machinery — infer a
signalling network with NicheNetPy, weight it with single-cell expression and
search for node knock-outs with EvoFlow:

1. **The IFN-β response.** Starting from the type I interferon receptor
   (`IFNAR1`/`IFNAR2`) and the interferon-stimulated genes it induces, find the
   intracellular mediators whose removal most reduces that response.
2. **The NK effector program.** Starting from the activating and inhibitory
   receptors NK cells use to sense other cells (`NCR1`, `KLRK1`, `TIGIT`, …),
   find the mediators that connect them to granule and cytokine effectors
   (`GNLY`, `NKG7`, `GZMB`, `IFNG`, `TNF`, …).

Along the way the notebook loads the `kang_2018` PBMC dataset with `pertpy`,
keeps the **IFN-β-stimulated** NK cells, builds the flow topology, runs the
genetic algorithm and interprets the four Pareto solutions. Control cells are
used only for the detection-rate comparison that shows how IFN-β rewrites the
network genes.

You can download the notebook here: {download}`use_case.ipynb`.

> The committed notebook is stored **pre-executed** with plotting resolution
> fixed at 96 DPI, so the repository stays small and the documentation renders
> without running NicheNet, `pertpy` or Docker during the build.
