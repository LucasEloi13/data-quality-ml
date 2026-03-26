#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except Exception:
    MATPLOTLIB_AVAILABLE = False

EPS = 1e-9

METRIC_ID_TO_NAME: Dict[int, str] = {
    4: "rows_ingestion_volume",
    1: "other_sellers_added_volume",
    13: "missing_response_rate",
    10: "request_success_rate",
    14: "request_error_mix",
    17: "update_interval_tolerance",
    2: "sellers_ingestion_volume",
    18: "availability_ratio_drop",
    19: "brand_diversity_ratio",
    20: "breadcrumb_missing_rate",
    21: "rating_missmatch",
    7: "name_missing_rate",
    5: "price_missing_rate",
    16: "match_rate",
    3: "plp_url_ingestion_volume",
    6: "response_latency_p95",
    12: "sku_missing_rate",
    11: "response_latency_p50",
    15: "instock_price_zero_rate",
    9: "brand_missing_rate",
    8: "images_missing_rate",
    22: "matching_name_anomaly_rate",
    23: "matching_kit_anomaly_rate",
    24: "matching_brand_anomaly_rate",
    25: "matching_price_anomaly_rate",
    26: "sellername_missing_rate",
    27: "sellercnpj_missing_rate",
    28: "sellerstate_missing_rate",
    29: "sellercity_missing_rate",
    30: "sellerzipcode_missing_rate",
    31: "sellercountry_missing_rate",
    32: "sellersocialreason_missing_rate",
    33: "currency_change_rate",
    34: "seller_diversity",
    35: "other_sellers_per_product_p50",
    36: "other_sellers_per_product_p90",
}

VOLUME_METRIC_IDS = {1, 2, 3, 4}

BASE_NUMERIC_COLS = [
    "rawValue",
    "rawLast",
    "avg3d",
    "avg7d",
    "avg15d",
    "avg30d",
    "stddev7d",
    "stddev30d",
]


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


def read_raw(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", dtype={"dimensions": "string"}, low_memory=False)
    for c in ["metricId", "crawlJobId", "organizationId", "sourceId"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in BASE_NUMERIC_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["metricId", "crawlJobId", "organizationId", "sourceId"]).copy()
    df[["metricId", "crawlJobId", "organizationId", "sourceId"]] = (
        df[["metricId", "crawlJobId", "organizationId", "sourceId"]].astype(int)
    )
    df["metricName"] = df["metricId"].map(METRIC_ID_TO_NAME).fillna("unknown_metric")
    return df


def aggregate_dimensions(df: pd.DataFrame) -> pd.DataFrame:
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
    out = out.drop(columns=to_drop).sort_values(["sourceId", "organizationId", "crawlJobId", "metricId"])
    return out


def add_metric_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d = d.sort_values(["sourceId", "organizationId", "metricId", "crawlJobId"])

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

    d["missing_fields_count"] = d[["rawValue", "rawLast", "avg3d", "avg7d", "avg15d", "avg30d", "stddev7d", "stddev30d"]].isna().sum(axis=1)

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

    # Clip very large ratios so one degenerate signal cannot dominate the score.
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
    d["is_metric_anomaly"] = d["row_score_robust_z"] > 3.5

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


def add_group_rolling_robust_z(
    df: pd.DataFrame, group_cols: List[str], order_col: str, feature_cols: List[str], window: int = 20
) -> pd.DataFrame:
    out = df.sort_values(group_cols + [order_col]).copy()

    for c in feature_cols:
        z_col = f"{c}_hist_rz"
        out[z_col] = np.nan

    for _, idx in out.groupby(group_cols).groups.items():
        sub = out.loc[idx].sort_values(order_col)
        for c in feature_cols:
            med = sub[c].shift(1).rolling(window, min_periods=6).median()
            q1 = sub[c].shift(1).rolling(window, min_periods=6).quantile(0.25)
            q3 = sub[c].shift(1).rolling(window, min_periods=6).quantile(0.75)
            scale = np.maximum(((q3 - q1).abs() / 1.349), 1.0)
            rz = (sub[c] - med) / scale
            rz = rz.clip(lower=-50, upper=50)
            out.loc[sub.index, f"{c}_hist_rz"] = rz

    return out


def build_crawljob_level(metric_df: pd.DataFrame) -> pd.DataFrame:
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
        out.get("request_success_rate", np.nan).fillna(0) + out.get("request_error_mix", np.nan).fillna(0) - 1.0
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
    if present_missing_cols:
        out["missingness_bundle_mean"] = out[present_missing_cols].mean(axis=1, skipna=True)
    else:
        out["missingness_bundle_mean"] = np.nan

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

    out = add_group_rolling_robust_z(
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
    out["is_crawljob_anomaly"] = out["crawljob_score_robust_z"] > 3.5

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


def build_explainability_outputs(
    metric_df: pd.DataFrame,
    crawljob_df: pd.DataFrame,
    output_dir: Path,
    top_metrics_per_job: int = 8,
    max_plots: int = 30,
) -> None:
    anomalies = crawljob_df[crawljob_df["is_crawljob_anomaly"]].copy()
    anomalies = anomalies.sort_values("crawljob_anomaly_score_raw", ascending=False)

    key_cols = ["sourceId", "organizationId", "crawlJobId"]
    merged = metric_df.merge(anomalies[key_cols], on=key_cols, how="inner")
    merged = merged.sort_values(key_cols + ["row_anomaly_score_raw"], ascending=[True, True, True, False])

    grouped = merged.groupby(key_cols, as_index=False)["row_anomaly_score_raw"].sum()
    grouped = grouped.rename(columns={"row_anomaly_score_raw": "sum_metric_score_in_job"})
    merged = merged.merge(grouped, on=key_cols, how="left")
    merged["metric_contribution_pct"] = (
        merged["row_anomaly_score_raw"] / np.maximum(merged["sum_metric_score_in_job"], EPS)
    )

    metric_detail = merged.groupby(key_cols, group_keys=False).head(top_metrics_per_job).copy()
    metric_detail = metric_detail[
        key_cols
        + [
            "metricId",
            "metricName",
            "rawValue",
            "rawLast",
            "avg7d",
            "avg30d",
            "row_anomaly_score_raw",
            "row_score_robust_z",
            "is_metric_anomaly",
            "metric_contribution_pct",
            "metric_anomaly_drivers",
        ]
    ]
    metric_detail = metric_detail.sort_values(key_cols + ["row_anomaly_score_raw"], ascending=[True, True, True, False])
    metric_detail.to_csv(output_dir / "crawljob_anomaly_metric_contributions.csv", index=False)

    top_metric_strings = (
        metric_detail.groupby(key_cols)
        .apply(
            lambda x: " | ".join(
                [
                    f"{r.metricName} ({r.metric_contribution_pct * 100:.1f}%)"
                    for _, r in x.sort_values("row_anomaly_score_raw", ascending=False).head(5).iterrows()
                ]
            ),
            include_groups=False,
        )
        .reset_index(name="top_metric_contributors")
    )

    overview = anomalies.merge(top_metric_strings, on=key_cols, how="left")
    overview_cols = key_cols + [
        "crawljob_anomaly_score_raw",
        "crawljob_score_robust_z",
        "metric_count",
        "metric_anomaly_count",
        "metric_anomaly_ratio",
        "crawljob_anomaly_drivers",
        "top_metric_contributors",
    ]
    overview = overview[overview_cols]
    overview.to_csv(output_dir / "crawljob_anomaly_explainability.csv", index=False)

    report_lines: List[str] = []
    report_lines.append("# Relatorio de Anomalias por Crawl Job")
    report_lines.append("")
    report_lines.append(f"Total de crawl jobs anomalos: {len(overview)}")
    report_lines.append("")
    report_lines.append("## Top alarmes")
    report_lines.append("")
    for _, row in overview.head(20).iterrows():
        report_lines.append(
            f"- source={int(row.sourceId)} org={int(row.organizationId)} crawlJob={int(row.crawlJobId)} | "
            f"score={row.crawljob_anomaly_score_raw:.3f} | z={row.crawljob_score_robust_z:.3f}"
        )
        report_lines.append(f"  drivers: {row.crawljob_anomaly_drivers}")
        report_lines.append(f"  metricas: {row.top_metric_contributors}")
    report_lines.append("")

    plots_note = "Graficos nao gerados (matplotlib indisponivel)."
    if MATPLOTLIB_AVAILABLE:
        plots_dir = output_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        top_for_plot = overview.head(min(max_plots, len(overview)))
        if not top_for_plot.empty:
            fig, ax = plt.subplots(figsize=(12, 6))
            labels = [
                f"s{int(r.sourceId)}-o{int(r.organizationId)}-c{int(r.crawlJobId)}"
                for _, r in top_for_plot.iterrows()
            ]
            ax.bar(labels, top_for_plot["crawljob_anomaly_score_raw"].to_numpy(dtype=float), color="#3366CC")
            ax.set_title("Top crawl jobs anomalos")
            ax.set_ylabel("crawljob_anomaly_score_raw")
            ax.tick_params(axis="x", labelrotation=75)
            fig.tight_layout()
            fig.savefig(plots_dir / "top_crawljob_anomalies.png", dpi=150)
            plt.close(fig)

        for _, row in top_for_plot.iterrows():
            key_mask = (
                (metric_detail["sourceId"] == row["sourceId"])
                & (metric_detail["organizationId"] == row["organizationId"])
                & (metric_detail["crawlJobId"] == row["crawlJobId"])
            )
            top_m = metric_detail[key_mask].sort_values("row_anomaly_score_raw", ascending=False).head(8)
            if top_m.empty:
                continue

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.barh(
                top_m["metricName"].astype(str).to_list()[::-1],
                top_m["row_anomaly_score_raw"].to_numpy(dtype=float)[::-1],
                color="#D95F02",
            )
            ax.set_title(
                f"Contribuicao de metricas | s{int(row.sourceId)} o{int(row.organizationId)} c{int(row.crawlJobId)}"
            )
            ax.set_xlabel("row_anomaly_score_raw")
            fig.tight_layout()
            fig.savefig(
                plots_dir / f"contrib_s{int(row.sourceId)}_o{int(row.organizationId)}_c{int(row.crawlJobId)}.png",
                dpi=150,
            )
            plt.close(fig)

        plots_note = f"Graficos gerados em: {plots_dir}"

    report_lines.append("## Artefatos de interpretacao")
    report_lines.append("")
    report_lines.append("- crawljob_anomaly_explainability.csv")
    report_lines.append("- crawljob_anomaly_metric_contributions.csv")
    report_lines.append(f"- {plots_note}")
    report_lines.append("")
    (output_dir / "anomaly_report.md").write_text("\n".join(report_lines), encoding="utf-8")


def run_pipeline(input_file: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = read_raw(input_file)
    aggregated = aggregate_dimensions(raw)
    metric_level = add_metric_features(aggregated)
    crawljob_level = build_crawljob_level(metric_level)

    aggregated.to_csv(output_dir / "aggregated_metrics_no_dimensions.csv", index=False)

    metric_level.sort_values("row_anomaly_score_raw", ascending=False).to_csv(
        output_dir / "metric_anomalies.csv", index=False
    )

    crawljob_level.sort_values("crawljob_anomaly_score_raw", ascending=False).to_csv(
        output_dir / "crawljob_anomalies.csv", index=False
    )

    build_explainability_outputs(metric_level, crawljob_level, output_dir)

    summary = pd.DataFrame(
        [
            {
                "rows_raw": len(raw),
                "rows_after_dimension_aggregation": len(aggregated),
                "metric_anomalies": int(metric_level["is_metric_anomaly"].sum()),
                "crawljob_anomalies": int(crawljob_level["is_crawljob_anomaly"].sum()),
                "unique_source_org_pairs": int(
                    aggregated[["sourceId", "organizationId"]].drop_duplicates().shape[0]
                ),
            }
        ]
    )
    summary.to_csv(output_dir / "anomaly_summary.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Anomaly detection for crawl metrics by source + organization.")
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input raw metrics file (semicolon separated).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory where anomaly CSVs will be written.",
    )

    args = parser.parse_args()
    run_pipeline(args.input, args.output_dir)
