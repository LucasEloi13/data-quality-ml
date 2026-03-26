from typing import List

import numpy as np
import pandas as pd

from anomaly_pipeline.constants import (
    BASE_NUMERIC_COLS,
    CRAWLJOB_ANOMALY_THRESHOLD,
    EPS,
    METRIC_ANOMALY_THRESHOLD,
    VOLUME_METRIC_IDS,
)
from anomaly_pipeline.stats import robust_zscore, top_drivers


class DimensionsAggregator:
    def aggregate(self, df: pd.DataFrame) -> pd.DataFrame:
        keys = ["sourceId", "organizationId", "crawlJobId", "metricId"]
        grouped = df.groupby(keys, as_index=False)

        base = grouped[["metricName"]].first()

        means = grouped[BASE_NUMERIC_COLS].mean()
        means = means.rename(columns={c: f"{c}_mean" for c in BASE_NUMERIC_COLS})

        sums = grouped[BASE_NUMERIC_COLS].sum(min_count=1)
        sums = sums.rename(columns={c: f"{c}_sum" for c in BASE_NUMERIC_COLS})

        out = base.merge(means, on=keys, how="left").merge(sums, on=keys, how="left")

        volume_mask = out["metricId"].isin(VOLUME_METRIC_IDS)
        for c in BASE_NUMERIC_COLS:
            out[c] = np.where(volume_mask, out[f"{c}_sum"], out[f"{c}_mean"])

        to_drop = [f"{c}_mean" for c in BASE_NUMERIC_COLS] + [f"{c}_sum" for c in BASE_NUMERIC_COLS]
        return out.drop(columns=to_drop).sort_values(["sourceId", "organizationId", "crawlJobId", "metricId"])


class MetricFeatureEngineer:
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy().sort_values(["sourceId", "organizationId", "metricId", "crawlJobId"])

        for w in [3, 7, 15, 30]:
            avg_col = f"avg{w}d"
            d[f"resid_{w}d"] = d["rawValue"] - d[avg_col]
            d[f"pct_dev_{w}d"] = d[f"resid_{w}d"] / np.maximum(d[avg_col].abs(), 1.0)

        z7_scale = np.maximum(d["stddev7d"].abs(), np.maximum(d["avg7d"].abs() * 0.05, 1.0))
        z30_scale = np.maximum(d["stddev30d"].abs(), np.maximum(d["avg30d"].abs() * 0.05, 1.0))
        d["z_7d"] = d["resid_7d"] / z7_scale
        d["z_30d"] = d["resid_30d"] / z30_scale

        d["momentum_vs_last"] = (d["rawValue"] - d["rawLast"]) / np.maximum(d["rawLast"].abs(), 1.0)
        d["baseline_drift_3v30"] = (d["avg3d"] - d["avg30d"]) / np.maximum(d["avg30d"].abs(), 1.0)
        d["volatility_7d"] = d["stddev7d"] / np.maximum(d["avg7d"].abs(), 1.0)
        d["volatility_30d"] = d["stddev30d"] / np.maximum(d["avg30d"].abs(), 1.0)
        d["same_as_last"] = (d["rawValue"] == d["rawLast"]).astype(float)

        d["missing_fields_count"] = d[
            ["rawValue", "rawLast", "avg3d", "avg7d", "avg15d", "avg30d", "stddev7d", "stddev30d"]
        ].isna().sum(axis=1)

        group_keys = ["sourceId", "organizationId", "metricId"]
        g = d.groupby(group_keys, group_keys=False)["rawValue"]

        hist_median = g.transform(lambda s: s.shift(1).rolling(15, min_periods=5).median())
        hist_q1 = g.transform(lambda s: s.shift(1).rolling(15, min_periods=5).quantile(0.25))
        hist_q3 = g.transform(lambda s: s.shift(1).rolling(15, min_periods=5).quantile(0.75))
        hist_iqr = hist_q3 - hist_q1

        d["hist_median_15"] = hist_median
        d["hist_iqr_15"] = hist_iqr
        hist_scale = np.maximum((hist_iqr / 1.349).abs(), 1.0)
        d["hist_robust_z_15"] = (d["rawValue"] - hist_median) / hist_scale

        clip_cols = [
            "pct_dev_3d",
            "pct_dev_7d",
            "pct_dev_15d",
            "pct_dev_30d",
            "z_7d",
            "z_30d",
            "momentum_vs_last",
            "baseline_drift_3v30",
            "hist_robust_z_15",
            "volatility_7d",
            "volatility_30d",
        ]
        for c in clip_cols:
            d[c] = d[c].clip(lower=-20, upper=20)

        d["max_abs_stat_z"] = np.maximum.reduce(
            [
                d["z_7d"].abs().fillna(0).to_numpy(dtype=float),
                d["z_30d"].abs().fillna(0).to_numpy(dtype=float),
                d["hist_robust_z_15"].abs().fillna(0).to_numpy(dtype=float),
            ]
        )

        d["row_anomaly_score_raw"] = (
            0.42 * d["max_abs_stat_z"].fillna(0)
            + 0.20 * d["pct_dev_30d"].abs().fillna(0)
            + 0.14 * d["pct_dev_7d"].abs().fillna(0)
            + 0.12 * d["momentum_vs_last"].abs().fillna(0)
            + 0.07 * d["baseline_drift_3v30"].abs().fillna(0)
            + 0.03 * d["volatility_7d"].fillna(0)
            + 0.02 * d["missing_fields_count"].fillna(0)
        )

        combo_keys = ["sourceId", "organizationId"]
        d["row_score_robust_z"] = d.groupby(combo_keys, group_keys=False)["row_anomaly_score_raw"].transform(robust_zscore)
        d["is_metric_anomaly"] = d["row_score_robust_z"] > METRIC_ANOMALY_THRESHOLD

        driver_cols = [
            "max_abs_stat_z",
            "pct_dev_30d",
            "pct_dev_7d",
            "momentum_vs_last",
            "baseline_drift_3v30",
            "volatility_7d",
            "missing_fields_count",
        ]
        d["metric_anomaly_drivers"] = d.apply(lambda r: top_drivers(r, driver_cols, 3), axis=1)

        return d


class CrawlJobScorer:
    def _add_group_rolling_robust_z(
        self, df: pd.DataFrame, group_cols: List[str], order_col: str, feature_cols: List[str], window: int = 20
    ) -> pd.DataFrame:
        out = df.sort_values(group_cols + [order_col]).copy()

        for c in feature_cols:
            out[f"{c}_hist_rz"] = np.nan

        for _, idx in out.groupby(group_cols).groups.items():
            sub = out.loc[idx].sort_values(order_col)
            for c in feature_cols:
                med = sub[c].shift(1).rolling(window, min_periods=6).median()
                q1 = sub[c].shift(1).rolling(window, min_periods=6).quantile(0.25)
                q3 = sub[c].shift(1).rolling(window, min_periods=6).quantile(0.75)
                scale = np.maximum(((q3 - q1).abs() / 1.349), 1.0)
                rz = (sub[c] - med) / scale
                out.loc[sub.index, f"{c}_hist_rz"] = rz.clip(lower=-50, upper=50)

        return out

    def score(self, metric_df: pd.DataFrame) -> pd.DataFrame:
        keys = ["sourceId", "organizationId", "crawlJobId"]

        agg = metric_df.groupby(keys, as_index=False).agg(
            metric_count=("metricId", "nunique"),
            metric_anomaly_count=("is_metric_anomaly", "sum"),
            top_metric_score=("row_anomaly_score_raw", "max"),
        )

        top3 = (
            metric_df.sort_values("row_anomaly_score_raw", ascending=False)
            .groupby(keys)
            .head(3)
            .groupby(keys, as_index=False)["row_anomaly_score_raw"]
            .mean()
            .rename(columns={"row_anomaly_score_raw": "top3_metric_score_mean"})
        )
        agg = agg.merge(top3, on=keys, how="left")
        agg["metric_anomaly_ratio"] = agg["metric_anomaly_count"] / (agg["metric_count"] + EPS)

        raw_pivot = metric_df.pivot_table(index=keys, columns="metricName", values="rawValue", aggfunc="mean")
        raw_pivot.columns = [str(c) for c in raw_pivot.columns]
        raw_pivot = raw_pivot.reset_index()

        out = agg.merge(raw_pivot, on=keys, how="left")

        out["success_error_consistency"] = (
            out.get("request_success_rate", np.nan).fillna(0)
            + out.get("request_error_mix", np.nan).fillna(0)
            - 1.0
        ).abs()

        out["latency_spread_ratio"] = out.get("response_latency_p95", np.nan) / np.maximum(
            out.get("response_latency_p50", np.nan).abs(), 1.0
        )
        out["rows_per_seller"] = out.get("rows_ingestion_volume", np.nan) / np.maximum(
            out.get("sellers_ingestion_volume", np.nan).abs(), 1.0
        )
        out["other_sellers_intensity"] = out.get("other_sellers_added_volume", np.nan) / np.maximum(
            out.get("rows_ingestion_volume", np.nan).abs(), 1.0
        )

        missing_rate_cols = [
            "missing_response_rate",
            "breadcrumb_missing_rate",
            "name_missing_rate",
            "price_missing_rate",
            "sku_missing_rate",
            "brand_missing_rate",
            "images_missing_rate",
            "sellername_missing_rate",
            "sellercnpj_missing_rate",
            "sellerstate_missing_rate",
            "sellercity_missing_rate",
            "sellerzipcode_missing_rate",
            "sellercountry_missing_rate",
            "sellersocialreason_missing_rate",
        ]
        present_missing_cols = [c for c in missing_rate_cols if c in out.columns]
        out["missingness_bundle_mean"] = (
            out[present_missing_cols].mean(axis=1, skipna=True) if present_missing_cols else np.nan
        )

        out["quality_signal_match_rate"] = out.get("match_rate", np.nan)

        feature_cols = [
            "top_metric_score",
            "top3_metric_score_mean",
            "metric_anomaly_ratio",
            "success_error_consistency",
            "latency_spread_ratio",
            "rows_per_seller",
            "other_sellers_intensity",
            "missingness_bundle_mean",
            "quality_signal_match_rate",
        ]

        out = self._add_group_rolling_robust_z(
            out,
            group_cols=["sourceId", "organizationId"],
            order_col="crawlJobId",
            feature_cols=feature_cols,
            window=20,
        )

        hist_rz_cols = [f"{c}_hist_rz" for c in feature_cols]
        out[hist_rz_cols] = out[hist_rz_cols].replace([np.inf, -np.inf], np.nan).clip(lower=-12, upper=12)
        out["cross_feature_signal"] = out[hist_rz_cols].abs().mean(axis=1, skipna=True).fillna(0)

        out["crawljob_anomaly_score_raw"] = (
            0.45 * out["top3_metric_score_mean"].fillna(0)
            + 0.30 * out["cross_feature_signal"].fillna(0)
            + 0.25 * (out["metric_anomaly_ratio"].fillna(0) * 10.0)
        )

        out["crawljob_score_robust_z"] = out.groupby(["sourceId", "organizationId"], group_keys=False)[
            "crawljob_anomaly_score_raw"
        ].transform(robust_zscore)
        out["is_crawljob_anomaly"] = out["crawljob_score_robust_z"] > CRAWLJOB_ANOMALY_THRESHOLD

        driver_cols = [
            "top3_metric_score_mean",
            "cross_feature_signal",
            "metric_anomaly_ratio",
            "success_error_consistency_hist_rz",
            "latency_spread_ratio_hist_rz",
            "rows_per_seller_hist_rz",
            "missingness_bundle_mean_hist_rz",
        ]
        out["crawljob_anomaly_drivers"] = out.apply(lambda r: top_drivers(r, driver_cols, 4), axis=1)

        return out.sort_values(["sourceId", "organizationId", "crawlJobId"])
