#!/usr/bin/env python3
"""Map coverage JSON results onto package functions and methods."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path


def iter_function_records(package_dir: Path):
    for path in sorted(package_dir.glob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        module = ".".join(path.with_suffix("").parts)
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                yield {
                    "module": module,
                    "qualname": node.name,
                    "path": str(path),
                    "start": node.lineno,
                    "end": node.end_lineno,
                }
            elif isinstance(node, ast.ClassDef):
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        yield {
                            "module": module,
                            "qualname": f"{node.name}.{item.name}",
                            "path": str(path),
                            "start": item.lineno,
                            "end": item.end_lineno,
                        }


def classify_record(record, coverage_files):
    coverage = coverage_files.get(record["path"])
    if coverage is None:
        return "no-coverage-data", 0, 0

    lines = set(range(record["start"], record["end"] + 1))
    executed = lines & set(coverage.get("executed_lines", []))
    missing = lines & set(coverage.get("missing_lines", []))
    if not executed and not missing:
        return "no-executable-lines", len(executed), len(missing)
    if not executed:
        return "unexecuted", len(executed), len(missing)
    if missing:
        return "partial", len(executed), len(missing)
    return "executed", len(executed), len(missing)


def load_coverage(path: Path | None):
    if path is None:
        return {}
    with path.open() as fp:
        return json.load(fp).get("files", {})


def build_report(package_dir: Path, coverage_json: Path | None):
    coverage_files = load_coverage(coverage_json)
    report = []
    for record in iter_function_records(package_dir):
        status, executed_lines, missing_lines = classify_record(record, coverage_files)
        report.append(
            {
                **record,
                "status": status,
                "executed_lines": executed_lines,
                "missing_lines": missing_lines,
            }
        )
    return report


def print_text_report(report):
    summary = {}
    for record in report:
        summary[record["status"]] = summary.get(record["status"], 0) + 1
    print("Function/method audit summary:")
    for status in sorted(summary):
        print(f"  {status}: {summary[status]}")
    print(f"  total: {len(report)}")

    for status in ("unexecuted", "partial", "no-executable-lines", "no-coverage-data"):
        rows = [record for record in report if record["status"] == status]
        if not rows:
            continue
        print(f"\n{status}:")
        for record in rows:
            print(
                "{path}:{start}-{end} {qualname} "
                "executed_lines={executed_lines} missing_lines={missing_lines}".format(
                    **record
                )
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--package-dir",
        default="ssapy",
        type=Path,
        help="Package directory to audit.",
    )
    parser.add_argument(
        "--coverage-json",
        type=Path,
        help="Coverage JSON produced by pytest-cov or coverage json.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Write machine-readable JSON instead of text.",
    )
    args = parser.parse_args()

    report = build_report(args.package_dir, args.coverage_json)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print_text_report(report)


if __name__ == "__main__":
    main()
