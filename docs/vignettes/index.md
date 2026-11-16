# Vignettes

End-to-end examples that run the full R + Python workflow. They are designed
to be executed inside the provided container
({doc}`../container`), where R, NicheNet and EvoFlow are pre-installed.

```{toctree}
:maxdepth: 1

use_case
```

## Use case: NK-cell signalling knock-outs

{doc}`use_case` walks through a realistic analysis:

1. **R / NicheNet** — from a short table of receptors of interest and target
   genes, infer a weighted ligand-to-target signalling network and an
   annotation table.
2. **Expression data** — load the `kang_2018` PBMC dataset with `pertpy`,
   keep the NK cells and build the samples × genes matrix for percolation.
3. **EvoFlow** — prepare the flow network, run the genetic algorithm and
   inspect the predicted knock-out sets.
4. **Interpretation** — compare the balanced, flow, cost and redundancy
   solutions and relate them to the biology.

You can download the notebook here: {download}`use_case.ipynb`.

> The committed notebook is stored **without outputs** and with plotting
> resolution fixed at 96 DPI, so the repository stays small. Run it in the
> container and save it with outputs before publishing a release.
