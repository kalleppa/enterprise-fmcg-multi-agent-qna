from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CHART_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "generated"
    / "charts"
)


AnalysisStatus = Literal[
    "success",
    "clarification",
    "unsupported",
    "error",
]


@dataclass
class AnalysisResponse:
    """Standard response returned by the coding agent."""

    status: AnalysisStatus
    operation: str
    message: str
    results: dict[str, Any] = field(
        default_factory=dict
    )
    chart_path: str | None = None
    assumptions: list[str] = field(
        default_factory=list
    )
    errors: list[str] = field(
        default_factory=list
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CodingAnalysisAgent:
    """
    Perform controlled calculations on approved structured data.

    Arbitrary Python execution, eval, exec, subprocesses and network
    access are intentionally not supported.
    """

    SUPPORTED_OPERATIONS = {
        "summary",
        "correlation",
        "percentage_change",
        "chart",
    }

    SUPPORTED_CHART_TYPES = {
        "bar",
        "line",
    }

    def analyze(
        self,
        rows: list[dict[str, Any]],
        operation: str,
        x_column: str | None = None,
        y_column: str | None = None,
        second_y_column: str | None = None,
        group_by: str | None = None,
        chart_type: str | None = None,
        aggregation: str = "sum",
        title: str | None = None,
    ) -> AnalysisResponse:
        """Run one allowlisted analytical operation."""

        normalized_operation = (
            operation.strip().lower()
        )

        if normalized_operation not in (
            self.SUPPORTED_OPERATIONS
        ):
            return AnalysisResponse(
                status="unsupported",
                operation=normalized_operation,
                message=(
                    "Unsupported coding operation. Supported "
                    "operations are summary, correlation, "
                    "percentage_change, and chart."
                ),
            )

        if not rows:
            return AnalysisResponse(
                status="clarification",
                operation=normalized_operation,
                message=(
                    "No structured-data rows were provided "
                    "for analysis."
                ),
            )

        dataframe = pd.DataFrame(rows)

        if dataframe.empty:
            return AnalysisResponse(
                status="clarification",
                operation=normalized_operation,
                message=(
                    "The provided structured-data result "
                    "is empty."
                ),
            )

        try:
            if normalized_operation == "summary":
                return self._summarize(
                    dataframe
                )

            if normalized_operation == "correlation":
                return self._calculate_correlation(
                    dataframe=dataframe,
                    first_column=y_column,
                    second_column=second_y_column,
                )

            if normalized_operation == (
                "percentage_change"
            ):
                return self._calculate_percentage_change(
                    dataframe=dataframe,
                    x_column=x_column,
                    y_column=y_column,
                )

            return self._create_chart(
                dataframe=dataframe,
                x_column=x_column,
                y_column=y_column,
                group_by=group_by,
                chart_type=chart_type,
                aggregation=aggregation,
                title=title,
            )

        except (
            KeyError,
            TypeError,
            ValueError,
        ) as error:
            return AnalysisResponse(
                status="error",
                operation=normalized_operation,
                message=(
                    "The requested analysis could not "
                    "be completed."
                ),
                errors=[str(error)],
            )

    def _summarize(
        self,
        dataframe: pd.DataFrame,
    ) -> AnalysisResponse:
        """Return summary statistics for numeric columns."""

        numeric_dataframe = dataframe.select_dtypes(
            include="number"
        )

        if numeric_dataframe.empty:
            return AnalysisResponse(
                status="clarification",
                operation="summary",
                message=(
                    "The supplied data does not contain "
                    "numeric columns."
                ),
            )

        summaries: dict[str, dict[str, Any]] = {}

        for column in numeric_dataframe.columns:
            series = numeric_dataframe[
                column
            ].dropna()

            if series.empty:
                continue

            summaries[str(column)] = {
                "count": int(series.count()),
                "sum": round(
                    float(series.sum()),
                    4,
                ),
                "mean": round(
                    float(series.mean()),
                    4,
                ),
                "minimum": round(
                    float(series.min()),
                    4,
                ),
                "maximum": round(
                    float(series.max()),
                    4,
                ),
            }

        return AnalysisResponse(
            status="success",
            operation="summary",
            message=(
                f"Generated summary statistics for "
                f"{len(summaries)} numeric column(s)."
            ),
            results={
                "row_count": len(dataframe),
                "column_count": len(
                    dataframe.columns
                ),
                "numeric_summaries": summaries,
            },
        )

    def _calculate_correlation(
        self,
        dataframe: pd.DataFrame,
        first_column: str | None,
        second_column: str | None,
    ) -> AnalysisResponse:
        """Calculate Pearson correlation between two columns."""

        if not first_column or not second_column:
            return AnalysisResponse(
                status="clarification",
                operation="correlation",
                message=(
                    "Provide both numeric columns for "
                    "the correlation calculation."
                ),
            )

        self._require_columns(
            dataframe,
            [
                first_column,
                second_column,
            ],
        )

        analysis_data = pd.DataFrame(
            {
                first_column: pd.to_numeric(
                    dataframe[first_column],
                    errors="coerce",
                ),
                second_column: pd.to_numeric(
                    dataframe[second_column],
                    errors="coerce",
                ),
            }
        ).dropna()

        if len(analysis_data) < 2:
            return AnalysisResponse(
                status="clarification",
                operation="correlation",
                message=(
                    "At least two complete numeric rows "
                    "are required for correlation."
                ),
            )

        correlation = analysis_data[
            first_column
        ].corr(
            analysis_data[second_column]
        )

        if pd.isna(correlation):
            return AnalysisResponse(
                status="clarification",
                operation="correlation",
                message=(
                    "Correlation is undefined because one "
                    "of the selected columns has no variation."
                ),
            )

        return AnalysisResponse(
            status="success",
            operation="correlation",
            message=(
                f"Calculated Pearson correlation between "
                f"{first_column} and {second_column}."
            ),
            results={
                "first_column": first_column,
                "second_column": second_column,
                "method": "pearson",
                "correlation": round(
                    float(correlation),
                    6,
                ),
                "observations": len(
                    analysis_data
                ),
            },
        )

    def _calculate_percentage_change(
        self,
        dataframe: pd.DataFrame,
        x_column: str | None,
        y_column: str | None,
    ) -> AnalysisResponse:
        """
        Calculate percentage change between the first and last values.

        When x_column is provided, records are sorted by that column.
        """

        if not y_column:
            return AnalysisResponse(
                status="clarification",
                operation="percentage_change",
                message=(
                    "Provide the numeric column for "
                    "percentage-change analysis."
                ),
            )

        required_columns = [y_column]

        if x_column:
            required_columns.append(x_column)

        self._require_columns(
            dataframe,
            required_columns,
        )

        analysis_data = dataframe.copy()

        if x_column:
            analysis_data = (
                analysis_data.sort_values(
                    by=x_column
                )
            )

        numeric_series = pd.to_numeric(
            analysis_data[y_column],
            errors="coerce",
        ).dropna()

        if len(numeric_series) < 2:
            return AnalysisResponse(
                status="clarification",
                operation="percentage_change",
                message=(
                    "At least two numeric observations "
                    "are required."
                ),
            )

        starting_value = float(
            numeric_series.iloc[0]
        )

        ending_value = float(
            numeric_series.iloc[-1]
        )

        if starting_value == 0:
            return AnalysisResponse(
                status="clarification",
                operation="percentage_change",
                message=(
                    "Percentage change cannot be calculated "
                    "because the starting value is zero."
                ),
            )

        percentage_change = (
            (ending_value - starting_value)
            / abs(starting_value)
            * 100
        )

        return AnalysisResponse(
            status="success",
            operation="percentage_change",
            message=(
                f"Calculated the percentage change in "
                f"{y_column}."
            ),
            results={
                "x_column": x_column,
                "y_column": y_column,
                "starting_value": round(
                    starting_value,
                    4,
                ),
                "ending_value": round(
                    ending_value,
                    4,
                ),
                "absolute_change": round(
                    ending_value
                    - starting_value,
                    4,
                ),
                "percentage_change": round(
                    percentage_change,
                    4,
                ),
                "observations": len(
                    numeric_series
                ),
            },
        )

    def _create_chart(
        self,
        dataframe: pd.DataFrame,
        x_column: str | None,
        y_column: str | None,
        group_by: str | None,
        chart_type: str | None,
        aggregation: str,
        title: str | None,
    ) -> AnalysisResponse:
        """Create a controlled line or bar chart."""

        selected_chart_type = (
            chart_type or "bar"
        ).strip().lower()

        if (
            selected_chart_type
            not in self.SUPPORTED_CHART_TYPES
        ):
            return AnalysisResponse(
                status="unsupported",
                operation="chart",
                message=(
                    "Supported chart types are bar and line."
                ),
            )

        effective_x_column = (
            group_by or x_column
        )

        if (
            not effective_x_column
            or not y_column
        ):
            return AnalysisResponse(
                status="clarification",
                operation="chart",
                message=(
                    "Provide an x-axis column and a numeric "
                    "y-axis column."
                ),
            )

        self._require_columns(
            dataframe,
            [
                effective_x_column,
                y_column,
            ],
        )

        chart_data = dataframe[
            [
                effective_x_column,
                y_column,
            ]
        ].copy()

        chart_data[y_column] = pd.to_numeric(
            chart_data[y_column],
            errors="coerce",
        )

        chart_data = chart_data.dropna(
            subset=[y_column]
        )

        if chart_data.empty:
            return AnalysisResponse(
                status="clarification",
                operation="chart",
                message=(
                    "The selected y-axis column does not "
                    "contain numeric data."
                ),
            )

        if group_by:
            chart_data = self._aggregate_chart_data(
                dataframe=chart_data,
                group_column=effective_x_column,
                value_column=y_column,
                aggregation=aggregation,
            )

        else:
            chart_data = chart_data.sort_values(
                by=effective_x_column
            )

        CHART_DIRECTORY.mkdir(
            parents=True,
            exist_ok=True,
        )

        chart_id = uuid4().hex[:12]

        safe_name = self._safe_filename(
            title
            or (
                f"{y_column}_by_"
                f"{effective_x_column}"
            )
        )

        output_path = (
            CHART_DIRECTORY
            / f"{safe_name}_{chart_id}.png"
        )

        figure, axis = plt.subplots(
            figsize=(10, 6)
        )

        if selected_chart_type == "bar":
            axis.bar(
                chart_data[
                    effective_x_column
                ].astype(str),
                chart_data[y_column],
            )

        else:
            axis.plot(
                chart_data[
                    effective_x_column
                ].astype(str),
                chart_data[y_column],
                marker="o",
            )

        axis.set_xlabel(effective_x_column)
        axis.set_ylabel(y_column)

        axis.set_title(
            title
            or (
                f"{y_column} by "
                f"{effective_x_column}"
            )
        )

        axis.tick_params(
            axis="x",
            rotation=45,
        )

        figure.tight_layout()

        figure.savefig(
            output_path,
            dpi=150,
            bbox_inches="tight",
        )

        plt.close(figure)

        relative_path = output_path.relative_to(
            PROJECT_ROOT
        ).as_posix()

        return AnalysisResponse(
            status="success",
            operation="chart",
            message=(
                f"Generated a {selected_chart_type} chart "
                f"with {len(chart_data)} plotted point(s)."
            ),
            results={
                "chart_type": selected_chart_type,
                "x_column": effective_x_column,
                "y_column": y_column,
                "aggregation": (
                    aggregation
                    if group_by
                    else None
                ),
                "plotted_points": len(
                    chart_data
                ),
            },
            chart_path=relative_path,
        )

    @staticmethod
    def _aggregate_chart_data(
        dataframe: pd.DataFrame,
        group_column: str,
        value_column: str,
        aggregation: str,
    ) -> pd.DataFrame:
        """Aggregate data using an allowlisted operation."""

        normalized_aggregation = (
            aggregation.strip().lower()
        )

        allowed_aggregations = {
            "sum",
            "mean",
            "min",
            "max",
            "count",
        }

        if (
            normalized_aggregation
            not in allowed_aggregations
        ):
            raise ValueError(
                "Unsupported aggregation. Use sum, mean, "
                "min, max, or count."
            )

        grouped = (
            dataframe.groupby(
                group_column,
                dropna=False,
            )[value_column]
            .agg(normalized_aggregation)
            .reset_index()
        )

        return grouped.sort_values(
            by=group_column
        )

    @staticmethod
    def _require_columns(
        dataframe: pd.DataFrame,
        required_columns: list[str],
    ) -> None:
        """Raise an error when requested columns are missing."""

        missing_columns = [
            column
            for column in required_columns
            if column not in dataframe.columns
        ]

        if missing_columns:
            raise KeyError(
                "Missing required column(s): "
                + ", ".join(missing_columns)
            )

    @staticmethod
    def _safe_filename(
        value: str,
    ) -> str:
        """Create a filesystem-safe chart filename."""

        cleaned = re.sub(
            r"[^a-zA-Z0-9_-]+",
            "_",
            value.strip(),
        )

        cleaned = re.sub(
            r"_+",
            "_",
            cleaned,
        ).strip("_")

        return cleaned.lower() or "analysis_chart"


def main() -> None:
    """Run example approved analyses."""

    rows = [
        {
            "region": "East Region",
            "promotion_spend_inr": 500000,
            "actual_sales_lift_pct": 5.0,
        },
        {
            "region": "North Region",
            "promotion_spend_inr": 700000,
            "actual_sales_lift_pct": 8.0,
        },
        {
            "region": "South Region",
            "promotion_spend_inr": 1200000,
            "actual_sales_lift_pct": 12.0,
        },
        {
            "region": "West Region",
            "promotion_spend_inr": 1500000,
            "actual_sales_lift_pct": 16.0,
        },
    ]

    agent = CodingAnalysisAgent()

    responses = [
        agent.analyze(
            rows=rows,
            operation="summary",
        ),
        agent.analyze(
            rows=rows,
            operation="correlation",
            y_column="promotion_spend_inr",
            second_y_column="actual_sales_lift_pct",
        ),
        agent.analyze(
            rows=rows,
            operation="percentage_change",
            x_column="promotion_spend_inr",
            y_column="actual_sales_lift_pct",
        ),
        agent.analyze(
            rows=rows,
            operation="chart",
            x_column="region",
            y_column="actual_sales_lift_pct",
            chart_type="bar",
            title="Actual Sales Lift by Region",
        ),
    ]

    for response in responses:
        print("=" * 70)
        print(response.to_dict())


if __name__ == "__main__":
    main()