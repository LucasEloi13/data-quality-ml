from pathlib import Path

import pandas as pd

from anomaly_pipeline.constants import BASE_NUMERIC_COLS, METRIC_ID_TO_NAME


class RawDataReader:
    def read(self, path: Path) -> pd.DataFrame:
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
