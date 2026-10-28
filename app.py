from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import pandas as pd
from pydantic.dataclasses import dataclass
from pyfedappwrap.learning.app_types import BaseEvaluationApp
from pyfedappwrap.learning.run_runfig import AppConfig, AppInputConfig, AppOutputConfig
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from typing_extensions import override


@dataclass
class ComparativeClusteringBenchmarkConfig(AppConfig):
    id_column: str = "sample_id"
    truth_label_column: str = "true_label"
    sort_by: str = "silhouette"


@dataclass
class ComparativeClusteringBenchmarkInput(AppInputConfig):
    features: Any = None
    predictions: Any = None
    ground_truth: Any = None


@dataclass
class ComparativeClusteringBenchmarkOutput(AppOutputConfig):
    output: Any | None = None


class ComparativeClusteringBenchmarkApp(
    BaseEvaluationApp[
        ComparativeClusteringBenchmarkConfig,
        ComparativeClusteringBenchmarkInput,
        ComparativeClusteringBenchmarkOutput,
    ]
):
    def __init__(self):
        super().__init__()

    @staticmethod
    def safe_internal_metric(
            metric_name: str,
            X: pd.DataFrame,
            labels: pd.Series,
    ) -> Optional[float]:
        unique_labels = labels.nunique(dropna=True)
        if unique_labels < 2 or unique_labels >= len(labels):
            return None

        if metric_name == "silhouette":
            return float(silhouette_score(X, labels))
        if metric_name == "davies_bouldin":
            return float(davies_bouldin_score(X, labels))
        if metric_name == "calinski_harabasz":
            return float(calinski_harabasz_score(X, labels))

        raise ValueError(f"Unsupported metric: {metric_name}")

    @staticmethod
    def dataframe_with_id_column(df: pd.DataFrame, id_column: str) -> pd.DataFrame:
        if id_column in df.columns:
            return df.copy()

        index_name = df.index.name
        if index_name == id_column or (index_name is None and id_column == "index"):
            return df.reset_index().rename(columns={df.index.name or "index": id_column})

        raise ValueError(
            f"Missing id column '{id_column}'. It must exist either as a column or as the DataFrame index.")


    def run_evaluate(
            self,
            data: ComparativeClusteringBenchmarkInput,
    ) -> ComparativeClusteringBenchmarkOutput:
        features = data.features
        predictions = data.predictions
        truth = data.ground_truth

        id_column = self.config.id_column
        truth_label_column = self.config.truth_label_column
        sort_by = self.config.sort_by

        features_df = self.dataframe_with_id_column(features, id_column)
        predictions_df = self.dataframe_with_id_column(predictions, id_column)
        truth_df = self.dataframe_with_id_column(truth, id_column) if truth is not None else None

        prediction_columns = [col for col in predictions_df.columns if col != id_column]
        if not prediction_columns:
            raise ValueError(
                "Predictions input must contain at least one prediction column besides the id column.")

        X = features_df.drop(columns=[id_column])
        rows: list[dict[str, object]] = []

        for pred_col in prediction_columns:
            merged = features_df[[id_column]].merge(
                predictions_df[[id_column, pred_col]],
                on=id_column,
                how="inner",
            )

            if len(merged) != len(features_df):
                raise ValueError(
                    f"Prediction column '{pred_col}' does not align with all feature rows.")

            labels = merged[pred_col]
            row: dict[str, object] = {
                "model": pred_col,
                "n_samples": int(len(features_df)),
                "n_clusters": int(labels.nunique(dropna=True)),
                "silhouette": self.safe_internal_metric("silhouette", X, labels),
                "davies_bouldin": self.safe_internal_metric("davies_bouldin", X, labels),
                "calinski_harabasz": self.safe_internal_metric("calinski_harabasz", X, labels),
            }

            if truth is not None:
                if truth_label_column not in truth_df.columns:
                    raise ValueError(
                        f"Ground truth input must contain '{truth_label_column}'."
                    )

                truth_merged = predictions_df[[id_column, pred_col]].merge(
                    truth_df[[id_column, truth_label_column]],
                    on=id_column,
                    how="inner",
                    suffixes=("_pred", "_truth"),
                )
                if len(truth_merged) != len(predictions_df):
                    raise ValueError(
                        f"Ground truth does not align with all prediction rows for column '{pred_col}'."
                    )

                pred_col_merged = pred_col
                if pred_col_merged not in truth_merged.columns:
                    pred_col_merged = f"{pred_col}_pred"

                truth_label_col_merged = truth_label_column
                if truth_label_col_merged not in truth_merged.columns:
                    truth_label_col_merged = f"{truth_label_column}_truth"

                row["adjusted_rand"] = float(
                    adjusted_rand_score(
                        truth_merged[truth_label_col_merged],
                        truth_merged[pred_col_merged],
                    )
                )
                row["adjusted_mutual_info"] = float(
                    adjusted_mutual_info_score(
                        truth_merged[truth_label_col_merged],
                        truth_merged[pred_col_merged],
                    )
                )
            else:
                row["adjusted_rand"] = None
                row["adjusted_mutual_info"] = None

            rows.append(row)

        scores = pd.DataFrame(rows)
        if scores.empty:
            raise ValueError("No prediction columns found for benchmarking.")

        if sort_by is not None and sort_by != "":
            ascending = sort_by == "davies_bouldin"
            scores = scores.sort_values(
                by=sort_by,
                ascending=ascending,
                na_position="last",
            ).reset_index(drop=True)
        else:
            scores = scores.reset_index(drop=True)
        return ComparativeClusteringBenchmarkOutput(output=scores)

    @override
    def get_test_data(self) -> Optional[dict[str, Path]]:
        return {
            "features": Path("data/uci_dataset17_encoded.csv"),
            "predictions": Path("data/labels_gaus.csv"),
            "ground_truth": Path("data/labels_gaus.csv"),
        }

    @override
    def get_test_hyperparams(self) -> Optional[dict[str, Any]]:
        return {
            "id_column": "index",
            "truth_label_column": "label",
            "sort_by": "",
        }
