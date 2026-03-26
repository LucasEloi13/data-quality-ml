from pathlib import Path

import pandas as pd

from anomaly_pipeline.explainability import ExplainabilityReporter
from anomaly_pipeline.io_utils import RawDataReader
from anomaly_pipeline.processing import CrawlJobScorer, DimensionsAggregator, MetricFeatureEngineer


class AnomalyPipeline:
    def __init__(
        self,
        reader: RawDataReader | None = None,
        aggregator: DimensionsAggregator | None = None,
        metric_engineer: MetricFeatureEngineer | None = None,
        crawljob_scorer: CrawlJobScorer | None = None,
        reporter: ExplainabilityReporter | None = None,
    ):
        self.reader = reader or RawDataReader()
        self.aggregator = aggregator or DimensionsAggregator()
        self.metric_engineer = metric_engineer or MetricFeatureEngineer()
        self.crawljob_scorer = crawljob_scorer or CrawlJobScorer()
        self.reporter = reporter or ExplainabilityReporter()

    def run(self, input_file: Path, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)

        raw = self.reader.read(input_file)
        aggregated = self.aggregator.aggregate(raw)
        metric_level = self.metric_engineer.transform(aggregated)
        crawljob_level = self.crawljob_scorer.score(metric_level)

        aggregated.to_csv(output_dir / "aggregated_metrics_no_dimensions.csv", index=False)
        metric_level.sort_values("row_anomaly_score_raw", ascending=False).to_csv(
            output_dir / "metric_anomalies.csv", index=False
        )
        crawljob_level.sort_values("crawljob_anomaly_score_raw", ascending=False).to_csv(
            output_dir / "crawljob_anomalies.csv", index=False
        )

        self.reporter.build(metric_level, crawljob_level, output_dir)

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
