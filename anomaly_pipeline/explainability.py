from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from anomaly_pipeline.constants import EPS


class ExplainabilityReporter:
    def __init__(self, top_metrics_per_job: int = 8):
        self.top_metrics_per_job = top_metrics_per_job

    def build(self, metric_df: pd.DataFrame, crawljob_df: pd.DataFrame, output_dir: Path) -> None:
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

        metric_detail = merged.groupby(key_cols, group_keys=False).head(self.top_metrics_per_job).copy()
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
        metric_detail = metric_detail.sort_values(
            key_cols + ["row_anomaly_score_raw"], ascending=[True, True, True, False]
        )
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

        self._write_markdown_report(overview, output_dir)
        self._write_plot_note(output_dir)

    def _write_markdown_report(self, overview: pd.DataFrame, output_dir: Path) -> None:
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

        plots_note = "Graficos removidos do pipeline."

        report_lines.append("## Artefatos de interpretacao")
        report_lines.append("")
        report_lines.append("- crawljob_anomaly_explainability.csv")
        report_lines.append("- crawljob_anomaly_metric_contributions.csv")
        report_lines.append(f"- {plots_note}")
        report_lines.append("")

        (output_dir / "anomaly_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    def _write_plot_note(self, output_dir: Path) -> None:
        plots_dir = output_dir / "plots"
        if plots_dir.exists() and plots_dir.is_dir():
            for p in plots_dir.glob("*"):
                if p.is_file():
                    p.unlink()
            plots_dir.rmdir()
