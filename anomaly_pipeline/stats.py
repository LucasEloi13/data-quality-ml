from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd

from anomaly_pipeline.constants import EPS


def robust_zscore(series: pd.Series) -> pd.Series:
    med = np.nanmedian(series.to_numpy(dtype=float))
    mad = np.nanmedian(np.abs(series.to_numpy(dtype=float) - med))
    if np.isnan(mad) or mad < EPS:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - med) / (1.4826 * mad + EPS)


def top_drivers(row: pd.Series, candidates: Iterable[str], top_k: int = 3) -> str:
    scored: List[Tuple[str, float]] = []
    for c in candidates:
        val = row.get(c, np.nan)
        if pd.isna(val):
            continue
        scored.append((c, float(abs(val))))
    scored.sort(key=lambda x: x[1], reverse=True)
    return "|".join([f"{k}:{v:.3f}" for k, v in scored[:top_k]])
