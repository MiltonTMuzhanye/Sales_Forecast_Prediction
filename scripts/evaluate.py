"""Evaluate trained forecasting models and generate metric reports."""

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.sales_forecast_system.utils.config import Config
from src.sales_forecast_system.utils.logger import setup_logger


logger = setup_logger(__name__)


def generate_reports(
    results: dict,
    output_dir: str = "reports/metrics",
) -> None:
    """Generate CSV, JSON, and text metric reports."""

    if "comparison" not in results:
        raise ValueError(
            "Training results do not contain a model comparison."
        )

    comparison = results["comparison"]

    if not isinstance(comparison, pd.DataFrame):
        comparison = pd.DataFrame(comparison)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    csv_path = output_path / "model_comparison.csv"

    comparison.to_csv(
        csv_path,
        index=True,
        index_label="model",
    )

    json_path = output_path / "model_comparison.json"

    json_data = comparison.reset_index().to_dict(orient="records")

    with open(json_path, "w") as file:
        json.dump(json_data, file, indent=2)

    summary_path = output_path / "evaluation_summary.txt"

    best_model = comparison["rmse"].idxmin()

    with open(summary_path, "w") as file:
        file.write("SALES FORECASTING MODEL EVALUATION\n")
        file.write("=" * 60 + "\n\n")

        file.write(f"Best model by RMSE: {best_model}\n\n")

        file.write(
            "Model comparison:\n"
            "------------------------------------------------------------\n"
        )

        for model_name, row in comparison.iterrows():
            file.write(
                f"{model_name:<12} "
                f"RMSE={row['rmse']:.2f}  "
                f"MAE={row['mae']:.2f}  "
                f"MAPE={row['mape']:.2f}%  "
                f"sMAPE={row['smape']:.2f}%  "
                f"R2={row['r2']:.4f}\n"
            )

        file.write("\n")
        file.write(
            "Evaluation scope: Store 1 / Department 1\n"
        )
        file.write(
            "Metrics are calculated on the held-out test period.\n"
        )

    logger.info("Metrics written to %s", output_path)

    print("\nEvaluation reports generated:")
    print(f"  {csv_path}")
    print(f"  {json_path}")
    print(f"  {summary_path}")
    print(f"\nBest model by RMSE: {best_model}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate trained forecasting models"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
    )

    parser.add_argument(
        "--results",
        type=str,
        default="artifacts/trained_models/training_results.pkl",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports/metrics",
    )

    args = parser.parse_args()

    try:
        # Load configuration to verify it remains valid.
        Config(args.config)

        results = joblib.load(args.results)

        print("\nModel Performance Comparison:")
        print(results["comparison"])

        generate_reports(
            results,
            output_dir=args.output_dir,
        )

        return 0

    except Exception as exc:
        logger.error("Evaluation failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
