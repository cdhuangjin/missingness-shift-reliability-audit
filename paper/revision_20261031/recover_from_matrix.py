"""Recover class recall sufficient statistics from the frozen result matrix.

For each class c and environment e, d_ec=(r_c-t_ec)/(r_c+eps).
Since t_ec=r_c(1-d_ec)-eps*d_ec and the mean of t_ec is saved,
the environments provide a linear system for the shared reference recalls.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).parent
EPS = 1e-12
f = pd.read_csv(ROOT / 'results/multiclass_subgroup_extended/block_results.csv')
f = f[f.task_type.eq('multiclass')].copy()
rows, checks = [], []
for block, g in f.groupby('block', sort=True):
    assert len(g) == 13 and g.environment.nunique() == 13
    D = [json.loads(x) for x in g.class_degradation]
    labels = sorted(D[0])
    assert all(sorted(d) == labels for d in D)
    a = np.array([[d[k] for k in labels] for d in D])
    X = 1 - a
    y = g.target_macro_recall.to_numpy() * len(labels) + EPS * a.sum(axis=1)
    ref, _, rank, sv = np.linalg.lstsq(X, y, rcond=None)
    assert rank == len(labels)
    target = X * ref[None, :] - EPS * a
    residual = float(np.max(np.abs(X @ ref - y)))
    assert residual < 1e-9 and np.min(ref) >= -1e-9 and np.max(ref) <= 1+1e-9
    assert np.min(target) >= -1e-9 and np.max(target) <= 1+1e-9
    assert abs(ref.mean() - g.reference_macro_recall.iloc[0]) < 1e-9
    hidden_match = True
    for i, (_, row) in enumerate(g.iterrows()):
        worst_relative_idx = int(np.argmax(a[i]))
        flag = (row.reference_macro_auroc-row.target_macro_auroc <= 0.03 and ref[worst_relative_idx]-target[i,worst_relative_idx] >= 0.10)
        hidden_match &= bool(flag) == bool(row.hidden_failure)
        for j, label in enumerate(labels):
            rows.append({'block':block,'dataset':row.dataset,'model':row.model,'seed':row.seed,'environment':row.environment,
                         'class':label,'reference_recall':ref[j],'target_recall':target[i,j],
                         'absolute_drop':ref[j]-target[i,j],'relative_drop':a[i,j]})
    assert hidden_match
    checks.append({'block':block,'n_classes':len(labels),'rank':int(rank),'condition_number':float(sv[0]/sv[-1]),'max_residual':residual,'hidden_flags_match':hidden_match})
out = pd.DataFrame(rows)
assert len(checks) == 36
validation = []
for checkpath in (OUT / 'recovery').glob('*_verification.json'):
    check = json.loads(checkpath.read_text(encoding='utf-8'))
    if not check['reproduced']: continue
    m = pd.read_csv(checkpath.with_name(checkpath.name.replace('_verification.json','_class_metrics.csv')), dtype={'class':str})
    merged = m.merge(out, on=['dataset','model','seed','environment','class'], validate='one_to_one', suffixes=('_direct','_recovered'))
    assert len(merged) == len(m)
    err = max(float((merged[c+'_direct']-merged[c+'_recovered']).abs().max()) for c in ['reference_recall','target_recall','absolute_drop'])
    assert err < 1e-9
    validation.append({'file':checkpath.name,'max_direct_difference':err})
out.to_csv(OUT / 'recovered_class_recalls.csv', index=False)
pd.DataFrame(checks).to_csv(OUT / 'recovery_system_checks.csv', index=False)
(OUT / 'recovery_validation.json').write_text(json.dumps({'blocks':len(checks),'direct_comparisons':validation,'note':'Algebraic recovery from saved relative metrics and target macro recall; precision depends on numerical conditioning.'},indent=2),encoding='utf-8')
print(json.dumps({'blocks':len(checks),'max_condition':max(x['condition_number'] for x in checks),'direct_comparisons':validation},indent=2))
