# Vignettes

End-to-end examples that run the full NicheNet + EvoFlow workflow. They are
designed to be executed inside the provided container
({doc}`../container`), where EvoFlow, NicheNetPy and pertpy are pre-installed.

```{toctree}
:maxdepth: 1

use_case
```

## Use case: IFN-β-stimulated NK cells

{doc}`use_case` walks through a realistic analysis:

1. **NicheNetPy** — from an IFN-β-focused receptor panel (IFNAR1/2, IL-2/15
   receptor, NKG2D, IL-12 receptor) and NK transcriptional/effector targets,
   infer a weighted ligand-to-target signalling network and an annotation
   table.
2. **Expression data** — load the `kang_2018` PBMC dataset with `pertpy`,
   keep the **IFN-β-stimulated** NK cells and build the samples × genes matrix
   for percolation. The control cells are used only for the detection-rate
   comparison.
3. **EvoFlow** — prepare the flow network, run the genetic algorithm and
   inspect the predicted knock-out sets.
4. **Interpretation** — compare the balanced, flow, cost and redundancy
   solutions and relate them to the IFN-β response and NK effector program.

You can download the notebook here: {download}`use_case.ipynb`.

> The committed notebook is stored **without outputs** and with plotting
> resolution fixed at 96 DPI, so the repository stays small. Run it in the
> container and save it with outputs before publishing a release.
