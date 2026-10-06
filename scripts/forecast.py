"""Generate sales forecasts and save them as JSON or CSV."""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.sales_forecast_system.pipelines.forecasting_pipeline import (
    ForecastingPipeline,
)
from src.sales_forecast_system.utils.config import Config
from src.sales_forecast_system.utils.logger import setup_logger


logger = setup_logger(__name__)


def save_forecast(result: dict, output_path: str) -> None:
    """Save forecast results as JSON or CSV based on file extension."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix.lower() == ".csv":
        forecast_df = pd.DataFrame(
            {
                "store_id": result["store_id"],
                "department_id": result["department_id"],
                "model": result["model"],
                "date": result["dates"],
                "prediction": result["predictions"],
            }
        )

        forecast_df.to_csv(path, index=False)

    elif path.suffix.lower() == ".json":
        with open(path, "w") as file:
            json.dump(result, file, indent=2, default=str)

    else:
        raise ValueError(
            "Unsupported output format. Use a .csv or .json file."
        )

    logger.info("Forecast saved to %s", path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate sales forecasts"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
    )

    parser.add_argument(
        "--store",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--dept",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--periods",
        type=int,
        default=12,
    )

    parser.add_argument(
        "--model",
        type=str,
        default="prophet",
        choices=["prophet", "xgboost", "lightgbm", "hybrid"],
    )

    parser.add_argument(
        "--output",
        type=str,
        help="Output file path (.csv or .json)",
    )

    args = parser.parse_args()

    if args.periods < 1:
        logger.error("--periods must be greater than zero")
        return 1

    try:
        config = Config(args.config)
        pipeline = ForecastingPipeline(config)

        result = pipeline.forecast(
            args.store,
            args.dept,
            args.periods,
            args.model,
        )

        if args.output:
            save_forecast(result, args.output)
        else:
            print(json.dumps(result, indent=2, default=str))

        return 0

    except Exception as exc:
        logger.error("Forecast failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
