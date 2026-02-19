"""Compare solver benchmark matrices and enforce slowdown budget."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ComparisonRow:
    """Comparison row for one shared benchmark case."""

    case_id: str
    baseline_ms: float
    candidate_ms: float
    slowdown_pct: float


def _load_cases(path: Path) -> dict[str, dict[str, object]]:
    """Load benchmark JSON and index cases by case_id.

    Args:
        path: Path to benchmark matrix JSON.

    Returns:
        Mapping from case id to case payload.
    """
    payload = json.loads(path.read_text())
    cases = payload.get("cases", [])
    indexed: dict[str, dict[str, object]] = {}
    for case in cases:
        case_id = str(case["case_id"])
        indexed[case_id] = dict(case)
    return indexed


def _compare(
    *,
    baseline: dict[str, dict[str, object]],
    candidate: dict[str, dict[str, object]],
) -> list[ComparisonRow]:
    """Compare median runtime per shared case.

    Args:
        baseline: Baseline benchmark mapping by case id.
        candidate: Candidate benchmark mapping by case id.

    Returns:
        Comparison rows for all shared cases.
    """
    shared_case_ids = sorted(set(baseline).intersection(candidate))
    rows: list[ComparisonRow] = []
    for case_id in shared_case_ids:
        baseline_ms = float(baseline[case_id]["steady_median_ms"])
        candidate_ms = float(candidate[case_id]["steady_median_ms"])
        if baseline_ms <= 0.0:
            slowdown_pct = 0.0
        else:
            slowdown_pct = ((candidate_ms - baseline_ms) / baseline_ms) * 100.0
        rows.append(
            ComparisonRow(
                case_id=case_id,
                baseline_ms=baseline_ms,
                candidate_ms=candidate_ms,
                slowdown_pct=slowdown_pct,
            )
        )
    return rows


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Returns:
        Parsed command-line namespace.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--max-slowdown-pct", type=float, default=5.0)
    parser.add_argument(
        "--require-same-cases",
        action="store_true",
        help="Fail if baseline and candidate case sets are not identical.",
    )
    return parser.parse_args()


def main() -> None:
    """Compare baseline/candidate benchmarks and enforce slowdown threshold."""
    args = _parse_args()
    baseline_cases = _load_cases(args.baseline)
    candidate_cases = _load_cases(args.candidate)

    baseline_ids = set(baseline_cases)
    candidate_ids = set(candidate_cases)
    if args.require_same_cases and baseline_ids != candidate_ids:
        missing_in_candidate = sorted(baseline_ids - candidate_ids)
        missing_in_baseline = sorted(candidate_ids - baseline_ids)
        if missing_in_candidate:
            print("Missing cases in candidate:")
            for case_id in missing_in_candidate:
                print(f"  - {case_id}")
        if missing_in_baseline:
            print("Additional cases in candidate:")
            for case_id in missing_in_baseline:
                print(f"  - {case_id}")
        raise SystemExit(1)

    rows = _compare(baseline=baseline_cases, candidate=candidate_cases)
    print(
        "| Case | Baseline Median [ms] | Candidate Median [ms] | Slowdown [%] |\n"
        "| --- | ---: | ---: | ---: |"
    )
    for row in rows:
        print(
            f"| {row.case_id} | {row.baseline_ms:.3f} | "
            f"{row.candidate_ms:.3f} | {row.slowdown_pct:+.2f} |"
        )

    failures = [row for row in rows if row.slowdown_pct > float(args.max_slowdown_pct)]
    if failures:
        print(f"\nPerformance regression detected (> {args.max_slowdown_pct:.2f}%):")
        for row in failures:
            print(f"  - {row.case_id}: {row.slowdown_pct:+.2f}%")
        raise SystemExit(1)

    print(
        f"\nAll shared cases passed slowdown threshold "
        f"({args.max_slowdown_pct:.2f}%)."
    )


if __name__ == "__main__":
    main()
