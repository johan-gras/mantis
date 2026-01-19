#!/usr/bin/env python3
"""
Benchmark regression detection script.

Compares criterion benchmark results against a baseline and fails if any
benchmark regresses by more than the specified threshold.

Usage:
    python scripts/check_bench_regression.py --threshold 0.10
    python scripts/check_bench_regression.py --baseline benchmarks/results/main.json --current target/criterion
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional


# Default threshold: 10% regression allowed
DEFAULT_THRESHOLD = 0.10

# Criterion results directory
CRITERION_DIR = "target/criterion"


def parse_criterion_estimate(estimate_path: Path) -> Optional[float]:
    """Parse a criterion estimates.json file and return the mean time in nanoseconds."""
    try:
        with open(estimate_path) as f:
            data = json.load(f)
        # Criterion stores point_estimate in the estimates.json
        return data.get("mean", {}).get("point_estimate")
    except (json.JSONDecodeError, FileNotFoundError, KeyError):
        return None


def collect_criterion_results(criterion_dir: Path) -> dict[str, float]:
    """Collect all benchmark results from criterion output directory."""
    results = {}

    if not criterion_dir.exists():
        print(f"Warning: Criterion directory not found: {criterion_dir}")
        return results

    # Walk through criterion directory structure
    # Structure: target/criterion/<group>/<bench_name>/new/estimates.json
    for group_dir in criterion_dir.iterdir():
        if not group_dir.is_dir() or group_dir.name.startswith("."):
            continue

        for bench_dir in group_dir.iterdir():
            if not bench_dir.is_dir():
                continue

            # Check for new estimates
            estimates_path = bench_dir / "new" / "estimates.json"
            if estimates_path.exists():
                mean_ns = parse_criterion_estimate(estimates_path)
                if mean_ns is not None:
                    bench_name = f"{group_dir.name}/{bench_dir.name}"
                    results[bench_name] = mean_ns

    return results


def load_baseline(baseline_path: Path) -> dict[str, float]:
    """Load baseline benchmark results from JSON file."""
    try:
        with open(baseline_path) as f:
            data = json.load(f)

        # Support both flat format and nested format
        if "benchmarks" in data:
            # Nested format: {"benchmarks": {"name": {"mean_ns": value}}}
            return {
                name: bench["mean_ns"]
                for name, bench in data["benchmarks"].items()
            }
        else:
            # Flat format: {"name": value}
            return {k: v for k, v in data.items() if isinstance(v, (int, float))}
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Warning: Could not load baseline from {baseline_path}: {e}")
        return {}


def current_environment() -> dict[str, str]:
    """Capture the current environment for baseline compatibility checks."""
    import platform

    os_name = platform.system()
    return {
        "os": f"{os_name} {platform.release()}",
        "os_name": os_name,
        "cpu": platform.processor() or "unknown",
        "machine": platform.machine() or "unknown",
        "python": platform.python_version(),
    }


def env_signature(env: dict[str, str]) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Normalize environment info into comparable fields."""
    os_name = env.get("os_name")
    if not os_name:
        os_value = env.get("os", "")
        os_name = os_value.split(" ")[0] if os_value else None
    machine = env.get("machine")
    cpu = env.get("cpu")
    return os_name or None, machine or None, cpu or None


def environments_match(baseline_env: dict[str, str], current_env: dict[str, str]) -> bool:
    """Return True when baseline and current environments are compatible."""
    base_os, base_machine, base_cpu = env_signature(baseline_env)
    curr_os, curr_machine, curr_cpu = env_signature(current_env)

    if base_os and curr_os and base_os != curr_os:
        return False
    if base_machine and curr_machine and base_machine != curr_machine:
        return False
    if base_cpu and curr_cpu and base_cpu != curr_cpu:
        return False
    return True


def save_results(results: dict[str, float], output_path: Path, include_env: bool = True):
    """Save benchmark results to JSON file."""
    from datetime import datetime, timezone

    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "benchmarks": {
            name: {
                "mean_ns": mean_ns,
            }
            for name, mean_ns in sorted(results.items())
        }
    }

    if include_env:
        output["environment"] = current_environment()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved results to {output_path}")


def check_regression(
    baseline: dict[str, float],
    current: dict[str, float],
    threshold: float
) -> tuple[list[str], list[str], list[str]]:
    """
    Compare current results against baseline.

    Returns:
        (failures, warnings, improvements) - lists of messages
    """
    failures = []
    warnings = []
    improvements = []

    for name, curr_time in sorted(current.items()):
        base_time = baseline.get(name)

        if base_time is None:
            # New benchmark, no baseline
            continue

        if base_time == 0:
            # Avoid division by zero
            continue

        change = (curr_time - base_time) / base_time

        if change > threshold:
            # Regression beyond threshold
            failures.append(
                f"{name}: {change:+.1%} slower "
                f"({base_time/1e6:.3f}ms -> {curr_time/1e6:.3f}ms)"
            )
        elif change > threshold / 2:
            # Warning zone (half threshold)
            warnings.append(
                f"{name}: {change:+.1%} slower "
                f"({base_time/1e6:.3f}ms -> {curr_time/1e6:.3f}ms)"
            )
        elif change < -threshold:
            # Significant improvement
            improvements.append(
                f"{name}: {change:+.1%} faster "
                f"({base_time/1e6:.3f}ms -> {curr_time/1e6:.3f}ms)"
            )

    return failures, warnings, improvements


def main():
    parser = argparse.ArgumentParser(
        description="Check for benchmark regressions"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Regression threshold (default: {DEFAULT_THRESHOLD})"
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path("benchmarks/results/main.json"),
        help="Path to baseline results JSON"
    )
    parser.add_argument(
        "--current",
        type=Path,
        default=Path(CRITERION_DIR),
        help="Path to current criterion results directory"
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Save current results to this path"
    )
    parser.add_argument(
        "--fail-on-missing-baseline",
        action="store_true",
        help="Fail if baseline file doesn't exist"
    )
    parser.add_argument(
        "--ignore-env",
        action="store_true",
        help="Skip baseline environment compatibility checks"
    )

    args = parser.parse_args()

    # Collect current results
    print(f"Collecting results from {args.current}...")
    current = collect_criterion_results(args.current)

    if not current:
        print("No benchmark results found!")
        sys.exit(1)

    print(f"Found {len(current)} benchmarks")

    # Save if requested
    if args.save:
        save_results(current, args.save)

    # Load baseline
    baseline = load_baseline(args.baseline)

    if not baseline:
        if args.fail_on_missing_baseline:
            print(f"Error: No baseline found at {args.baseline}")
            sys.exit(1)
        else:
            print(f"No baseline found at {args.baseline}, skipping regression check")
            print("\nCurrent benchmark results:")
            for name, time_ns in sorted(current.items()):
                print(f"  {name}: {time_ns/1e6:.3f}ms")
            sys.exit(0)

    print(f"Loaded baseline with {len(baseline)} benchmarks")

    baseline_env = {}
    if isinstance(baseline, dict) and args.baseline.exists():
        try:
            with open(args.baseline) as f:
                baseline_json = json.load(f)
            baseline_env = baseline_json.get("environment", {})
        except (json.JSONDecodeError, FileNotFoundError):
            baseline_env = {}

    current_env = current_environment()

    if baseline_env and not args.ignore_env:
        if not environments_match(baseline_env, current_env):
            print(
                "Baseline environment does not match current run; "
                "skipping regression check."
            )
            print(f"  baseline: {baseline_env}")
            print(f"  current:  {current_env}")
            sys.exit(0)

    # Check for regressions
    failures, warnings, improvements = check_regression(
        baseline, current, args.threshold
    )

    # Report results
    if improvements:
        print("\nImprovements:")
        for msg in improvements:
            print(f"  {msg}")

    if warnings:
        print("\nWarnings (approaching threshold):")
        for msg in warnings:
            print(f"  {msg}")

    if failures:
        print(f"\nRegressions detected (>{args.threshold:.0%} slower):")
        for msg in failures:
            print(f"  {msg}")
        sys.exit(1)

    print(f"\nAll benchmarks within {args.threshold:.0%} threshold")
    sys.exit(0)


if __name__ == "__main__":
    main()
