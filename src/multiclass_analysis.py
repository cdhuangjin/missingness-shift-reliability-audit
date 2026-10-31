"""Statistical summaries for the multiclass/subgroup study."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon


def bootstrap_mean_ci(values, iterations: int = 2000, seed: int = 7301, confidence: float = 0.95):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(values.mean())
    if len(values) == 1:
        return mean, mean, mean
    rng = np.random.default_rng(seed)
    samples = values[rng.integers(0, len(values), size=(iterations, len(values)))].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return mean, float(np.quantile(samples, alpha)), float(np.quantile(samples, 1.0 - alpha))


def paired_wilcoxon(frame: pd.DataFrame, left: str, right: str) -> dict:
    values = frame[[left, right]].dropna()
    differences = values[left].to_numpy(dtype=float) - values[right].to_numpy(dtype=float)
    if len(differences) == 0:
        return {"n": 0, "effect_mean": float("nan"), "p_value": float("nan"), "statistic": float("nan")}
    if np.allclose(differences, 0.0):
        statistic, p_value = 0.0, 1.0
    else:
        statistic, p_value = wilcoxon(differences, alternative="two-sided", zero_method="wilcox")
    return {
        "n": int(len(differences)),
        "effect_mean": float(differences.mean()),
        "p_value": float(p_value),
        "statistic": float(statistic),
    }


def zero_reference_wilcoxon(values) -> dict:
    values = np.asarray(values, dtype=float)
    frame = pd.DataFrame({"left": values, "right": np.zeros(len(values), dtype=float)})
    return paired_wilcoxon(frame, "left", "right")


def adjust_bh(p_values: list[float] | np.ndarray) -> list[float]:
    values = np.asarray(p_values, dtype=float)
    result = np.full(values.shape, np.nan, dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        return result.tolist()
    indices = np.flatnonzero(finite)
    order = indices[np.argsort(values[finite])]
    ranked = values[order] * len(order) / np.arange(1, len(order) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    result[order] = np.clip(ranked, 0.0, 1.0)
    return result.tolist()


def parse_json_column(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    rows = []
    for _, row in frame.iterrows():
        payload = json.loads(row[column]) if isinstance(row[column], str) else {}
        for key, value in payload.items():
            record = row.drop(labels=[column]).to_dict()
            record["cell"] = key
            record["degradation"] = float(value)
            rows.append(record)
    return pd.DataFrame(rows)


def exposure_degradation_correlations(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in frame.iterrows():
        exposures = json.loads(row["exposure_by_class"])
        degradations = json.loads(row["class_degradation"])
        keys = sorted(set(exposures) & set(degradations))
        if len(keys) < 3:
            continue
        correlation = float(spearmanr([exposures[key] for key in keys], [degradations[key] for key in keys]).statistic)
        rows.append(
            {
                "dataset": row["dataset"],
                "model": row["model"],
                "seed": row["seed"],
                "block": row["block"],
                "environment": row["environment"],
                "correlation": correlation,
            }
        )
    return pd.DataFrame(rows)


def summarize_primary(frame: pd.DataFrame) -> pd.DataFrame:
    group_columns = ["task_type", "dataset", "model", "environment"]
    summaries = []
    for keys, block in frame.groupby(group_columns, sort=True):
        mean, lower, upper = bootstrap_mean_ci(block["class_spread"])
        row = dict(zip(group_columns, keys))
        row.update({"mean_class_spread": mean, "ci_lower": lower, "ci_upper": upper, "n_blocks": int(block["block"].nunique())})
        if block["task_type"].iloc[0] == "subgroup_binary":
            subgroup_mean, subgroup_lower, subgroup_upper = bootstrap_mean_ci(block["subgroup_gap"])
            row.update({"mean_subgroup_gap": subgroup_mean, "subgroup_ci_lower": subgroup_lower, "subgroup_ci_upper": subgroup_upper})
        summaries.append(row)
    return pd.DataFrame(summaries)
