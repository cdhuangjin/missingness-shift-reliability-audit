"""Analyze algebraically recovered class recalls without treating blocks as datasets."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
c = pd.read_csv(HERE / 'recovered_class_recalls.csv', dtype={'class':str})
f = pd.read_csv(ROOT / 'results/multiclass_subgroup_extended/block_results.csv')
f = f[f.task_type.eq('multiclass')].copy()
keys = ['block','dataset','model','seed','environment']
a = c.groupby(keys).agg(worst_absolute_drop=('absolute_drop','max'),max_drop=('absolute_drop','max'),min_drop=('absolute_drop','min')).reset_index()
a['absolute_spread'] = a.max_drop-a.min_drop
a = a.merge(f[keys+['reference_macro_auroc','target_macro_auroc','hidden_failure']],on=keys,validate='one_to_one')
a['auroc_drop'] = a.reference_macro_auroc-a.target_macro_auroc
a.to_csv(HERE / 'absolute_drop_by_block.csv',index=False)
rows=[]
for rate in (20,30,40):
    left=a[a.environment.eq(f'cc_{rate}')]
    right=a[a.environment.eq(f'matched_mcar_cc{rate}')]
    paired=left.merge(right,on=['block','dataset','model','seed'],suffixes=('_cc','_mcar'),validate='one_to_one')
    for metric in ('worst_absolute_drop','absolute_spread'):
        paired['effect']=paired[metric+'_cc']-paired[metric+'_mcar']
        for dataset,g in paired.groupby('dataset'):
            rows.append({'rate':rate,'metric':metric,'dataset':dataset,'n_blocks':len(g),'cc_mean':g[metric+'_cc'].mean(),'mcar_mean':g[metric+'_mcar'].mean(),'mean_contrast':g.effect.mean(),'positive_blocks':int((g.effect>0).sum()),'block_min':g.effect.min(),'block_max':g.effect.max()})
pd.DataFrame(rows).to_csv(HERE / 'absolute_contrasts_by_dataset.csv',index=False)
grid=[]
for env in ('cc_30','matched_mcar_cc30','reverse_cc_30'):
    q=a[a.environment.eq(env)]
    for u in (0.02,0.03,0.05):
        for r in (0.05,0.10,0.15):
            for dataset,g in q.groupby('dataset'):
                grid.append({'environment':env,'auroc_threshold':u,'absolute_worst_drop_threshold':r,'dataset':dataset,'count':int(((g.auroc_drop<=u)&(g.worst_absolute_drop>=r)).sum()),'n_blocks':len(g)})
pd.DataFrame(grid).to_csv(HERE / 'hidden_failure_absolute_threshold_grid.csv',index=False)
sens=[]
for threshold in (0.0,0.05,0.1,0.2):
    keep=c[c.reference_recall.ge(threshold)].copy()
    s=keep.groupby(keys).agg(cvs=('relative_drop',lambda x:x.max()-x.min()),classes_retained=('class','count')).reset_index()
    s=s[s.classes_retained.ge(2)]
    p=s[s.environment.eq('cc_30')].merge(s[s.environment.eq('matched_mcar_cc30')],on=['block','dataset','model','seed'],suffixes=('_cc','_mcar'),validate='one_to_one')
    p['effect']=p.cvs_cc-p.cvs_mcar
    for dataset,g in p.groupby('dataset'):
        sens.append({'minimum_reference_recall':threshold,'dataset':dataset,'n_blocks':len(g),'mean_contrast':g.effect.mean(),'min_classes_retained':int(g.classes_retained_cc.min()),'max_classes_retained':int(g.classes_retained_cc.max())})
pd.DataFrame(sens).to_csv(HERE / 'cvs_low_reference_sensitivity.csv',index=False)
print('Wrote block-level absolute metrics, dataset contrasts, threshold grid, and low-recall CVS sensitivity.')
