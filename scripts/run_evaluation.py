from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.evaluation.benchmark import (
    DEFAULT_OUTPUT_DIRECTORY,
    DEFAULT_QUESTIONS_PATH,
    BenchmarkRunner,
    load_scenarios,
    write_benchmark_report,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the enterprise FMCG "
            "multi-agent benchmark."
        )
    )

    parser.add_argument(
        "--questions",
        type=Path,
        default=DEFAULT_QUESTIONS_PATH,
        help="Path to the benchmark JSON file.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
        help="Directory for JSON and CSV reports.",
    )

    parser.add_argument(
        "--fail-under",
        type=float,
        default=70.0,
        help=(
            "Exit with code 1 when the scenario "
            "pass rate is below this percentage."
        ),
    )

    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    scenarios = load_scenarios(
        arguments.questions
    )

    runner = BenchmarkRunner()

    report = runner.run(
        scenarios
    )

    json_path, csv_path = (
        write_benchmark_report(
            report=report,
            output_directory=(
                arguments.output_dir
            ),
        )
    )

    summary = report["summary"]

    print("=" * 72)
    print("Enterprise FMCG Multi-Agent Benchmark")
    print("=" * 72)

    print(
        "Scenarios:",
        summary["total_scenarios"],
    )

    print(
        "Passed:",
        summary["passed_scenarios"],
    )

    print(
        "Failed:",
        summary["failed_scenarios"],
    )

    print(
        "Pass rate:",
        (
            f"{summary['scenario_pass_rate_pct']}"
            "%"
        ),
    )

    print(
        "Average turn score:",
        summary["average_turn_score"],
    )

    print(
        "Median latency:",
        summary["median_latency_ms"],
        "ms",
    )

    print(
        "P95 latency:",
        summary["p95_latency_ms"],
        "ms",
    )

    print()
    print("Category results:")

    for category, result in (
        report["categories"].items()
    ):
        print(
            f"- {category}: "
            f"{result['passed']}/"
            f"{result['scenario_count']} "
            f"({result['pass_rate_pct']}%)"
        )

    print()
    print("JSON report:", json_path)
    print("CSV report:", csv_path)

    if (
        summary["scenario_pass_rate_pct"]
        < arguments.fail_under
    ):
        print()
        print(
            "Benchmark pass rate is below "
            f"{arguments.fail_under}%."
        )

        sys.exit(1)


if __name__ == "__main__":
    main()