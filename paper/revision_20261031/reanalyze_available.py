"""Descriptive reanalysis of saved results; no refitting or invented recall values."""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.multiclass_analysis import adjust_bh, zero_reference_wilcoxon, paired_wilcoxon

OUT = Path(__file__).parent
source = ROOT / 'results/multiclass_subgroup_extended/block_results.csv'
frame = pd.read_csv(source)
assert len(frame) == 780 and not frame.duplicated(['block', 'environment']).any()
multi = frame[frame.task_type.eq('multiclass')]
rows = []
for rate in (20, 30, 40):
    lhs = multi[multi.environment.eq(f'cc_{rate}')][['dataset', 'block', 'class_spread']]
    rhs = multi[multi.environment.eq(f'matched_mcar_cc{rate}')][['block', 'class_spread']]
    pair = lhs.merge(rhs, on='block', validate='one_to_one', suffixes=('_cc', '_mcar'))
    assert len(pair) == 36
    pair['difference'] = pair.class_spread_cc - pair.class_spread_mcar
    for dataset, g in pair.groupby('dataset'):
        rows.append({'contrast':f'cc_{rate} - matched_mcar_cc{rate}', 'dataset':dataset,
                     'mean_effect':g.difference.mean(), 'n_blocks':len(g),
                     'positive_blocks':int((g.difference > 0).sum()),
                     'block_min':g.difference.min(), 'block_max':g.difference.max()})
pd.DataFrame(rows).to_csv(OUT / 'dataset_cvs_contrasts.csv', index=False)

def paired(a, b):
    left = multi[multi.environment.eq(a)][['block','class_spread']].rename(columns={'class_spread':'a'})
    right = multi[multi.environment.eq(b)][['block','class_spread']].rename(columns={'class_spread':'b'})
    return paired_wilcoxon(left.merge(right,on='block',validate='one_to_one'), 'a', 'b')['p_value']

primary = zero_reference_wilcoxon(multi[multi.environment.eq('cc_30')].class_spread)['p_value']
matched = paired('cc_30','matched_mcar_cc30')
reverse = paired('cc_30','reverse_cc_30')
families = {}
for group in ('adult', 'all'):
    subset = frame[(frame.task_type.eq('subgroup_binary')) & (frame.environment.eq('cc_30'))]
    if group != 'all': subset = subset[subset.dataset.eq(group)]
    subgroup = zero_reference_wilcoxon(subset.subgroup_gap)['p_value']
    raw = [primary, matched, reverse, subgroup]
    families[group] = {'names':['CVS versus zero','CVS cc30 versus matched','CVS cc30 versus reverse','subgroup gap versus zero'], 'raw':raw, 'adjusted':adjust_bh(raw), 'subgroup_n':len(subset)}
raw = [paired(f'cc_{r}', f'matched_mcar_cc{r}') for r in (20,30,40)]
families['intensity'] = {'names':['cc20','cc30','cc40'], 'raw':raw, 'adjusted':adjust_bh(raw)}
(OUT / 'bh_family_recalculation.json').write_text(json.dumps(families,indent=2,default=float),encoding='utf-8')

natural = ROOT / 'results/gate_natural_temporal_missingness/temporal_block_results.csv'
pd.read_csv(natural).to_csv(OUT / 'natural_temporal_observed_rows.csv',index=False)
manifest = {'input_sha256': {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (source,natural)}, 'notes':['CVS is relative degradation spread, not absolute recall drop.', 'Blocks share datasets; no within-dataset CI is asserted.', 'Natural temporal results use a separate BAF study, not the UCI matrix.']}
(OUT / 'analysis_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print('Wrote dataset contrasts, BH recalculation, natural-temporal rows, and input hashes.')
