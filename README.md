# Missingness-shift reliability audit: revision analyses

This repository contains the saved 780-row multiclass/subgroup result matrix and scripts used to reconstruct per-class recall quantities and rerun the revision's descriptive analyses. It also includes a separate BAF temporal summary used only for an audit of scope. The BAF analysis is not validation of the multiclass UCI estimands.

## Contents and reproducibility

The frozen matrix is `results/multiclass_subgroup_extended/block_results.csv`. It contains aggregate model results, not raw UCI records or model predictions. The three scripts are in `paper/revision_20261031/`; their layout is preserved to match the original project paths.

Install the dependencies in `requirements.txt`, then run from the repository root:

```bash
python paper/revision_20261031/recover_from_matrix.py
python paper/revision_20261031/analyze_recovered.py
python paper/revision_20261031/reanalyze_available.py
```

The first script solves a full-rank linear system for each multiclass model–seed block, checks that the recovered hidden-failure indicators match those in the frozen matrix, and writes per-class recall and system diagnostics. The second reports matched-rate MCAR contrasts, an absolute-drop threshold grid and low-reference-recall CVS sensitivity. The third reproduces BH test families and descriptive contrasts. These are analyses of existing outputs, not a full model-training pipeline. Original predictions, fitted models and raw datasets are not included.

The optional `recovery/` directory containing independent model rerun files is not included, so the script does not repeat those independent comparisons in this package. In the original project, four refits matched reconstructed recalls within 7.1e-14; one random-forest refit did not reproduce the frozen results. The result matrix does not establish full training reproducibility. Blocks share datasets, and uncertainty across only two or three datasets is substantial.

The absolute-worst-class threshold rule in the revision differs from the original rule, which selected the class with the largest relative degradation before evaluating its absolute drop. Counts from these rules must not be treated as identical.

The repository is publicly readable. No archival DOI or open-source licence is asserted here; consult the authors concerning permissions for reuse.
