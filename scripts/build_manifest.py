#!/usr/bin/env python
"""Build a trace-level manifest from an HDF5 matrix and companion metadata CSV."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import h5py

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data.io import write_manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Map every HDF5 tracing to patient/record metadata without assuming CODE-II's release layout"
    )
    parser.add_argument("--signals", type=Path, required=True)
    parser.add_argument("--dataset-key", required=True)
    parser.add_argument("--metadata-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--patient-column", default="patient_id")
    parser.add_argument("--record-column", default="record_id")
    parser.add_argument("--exam-column", default="exam_id")
    parser.add_argument("--tracing-column", default="tracing_index")
    parser.add_argument("--sampling-rate-column", default="sampling_rate")
    parser.add_argument("--split-column", default="split")
    parser.add_argument("--dataset-part-column", default="dataset_part")
    parser.add_argument("--amplitude-unit-column", default="amplitude_unit")
    parser.add_argument("--lead-names", nargs="+", required=True)
    parser.add_argument(
        "--force-split",
        choices=("train", "validation", "test"),
        help="Assign every row to one known official partition, e.g. CODE-II-test",
    )
    parser.add_argument(
        "--axis-order",
        choices=("records_samples_leads", "records_leads_samples"),
        required=True,
    )
    parser.add_argument("--path-relative-to", type=Path)
    args = parser.parse_args()

    with h5py.File(args.signals, "r") as handle:
        if args.dataset_key not in handle:
            raise ValueError(f"HDF5 dataset {args.dataset_key!r} not found")
        dataset = handle[args.dataset_key]
        if dataset.ndim != 3:
            raise ValueError(f"Expected record matrix with 3 dimensions, got {dataset.shape}")
        n_records = int(dataset.shape[0])

    with args.metadata_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        metadata = list(csv.DictReader(handle))
    if len(metadata) != n_records:
        raise ValueError(
            f"Metadata has {len(metadata)} rows but HDF5 dataset has {n_records} records"
        )
    required = {args.patient_column, args.record_column}
    missing = required - set(metadata[0])
    if missing:
        raise ValueError(f"Metadata is missing required columns: {sorted(missing)}")

    signal_path = args.signals.resolve()
    if args.path_relative_to:
        signal_path_text = str(signal_path.relative_to(args.path_relative_to.resolve()))
    else:
        signal_path_text = str(signal_path)
    rows = []
    for index, source in enumerate(metadata):
        rows.append(
            {
                "path": signal_path_text,
                "source_format": "hdf5",
                "patient_id": source[args.patient_column],
                "record_id": source[args.record_column],
                "exam_id": source.get(args.exam_column, ""),
                "tracing_index": source.get(args.tracing_column, ""),
                "dataset_key": args.dataset_key,
                "index": index,
                "sampling_rate": source.get(args.sampling_rate_column, ""),
                "lead_names": json.dumps(args.lead_names),
                "axis_order": args.axis_order,
                "amplitude_unit": source.get(args.amplitude_unit_column, ""),
                "official_split": args.force_split or source.get(args.split_column, ""),
                "dataset_part": source.get(args.dataset_part_column, ""),
            }
        )
    write_manifest(args.output, rows)
    print(f"Wrote {len(rows)} trace-level rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
