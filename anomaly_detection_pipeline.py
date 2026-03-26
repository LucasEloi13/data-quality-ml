#!/usr/bin/env python3
import argparse
from pathlib import Path

from anomaly_pipeline import AnomalyPipeline


def main() -> None:
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
    AnomalyPipeline().run(args.input, args.output_dir)


if __name__ == "__main__":
    main()
