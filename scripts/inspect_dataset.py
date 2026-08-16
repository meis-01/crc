#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data.inspection import inspect_manifest, write_inspection_outputs
from data.io import read_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect raw CODE-II recordings from a trace-level manifest")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--report-json", type=Path, default=Path("reports/dataset_inspection.json"))
    parser.add_argument("--issues-csv", type=Path, default=Path("reports/data_quality_issues.csv"))
    args = parser.parse_args()
    summary, issues = inspect_manifest(read_manifest(args.manifest), args.raw_root)
    write_inspection_outputs(summary, issues, args.report_json, args.issues_csv)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

