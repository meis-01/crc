#!/usr/bin/env python
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data.constants import CANONICAL_LEADS, normalize_target_lead
from data.dataset import decode_h5_string, invert_record


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot representative input and target leads by split")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--target-lead", default="V2")
    parser.add_argument("--output-dir", type=Path, default=Path("reports/figures"))
    parser.add_argument("--per-split", type=int, default=1)
    args = parser.parse_args()
    target = normalize_target_lead(args.target_lead)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.dataset, "r") as handle:
        selected: dict[str, list[int]] = {"train": [], "validation": [], "test": []}
        for index, split_raw in enumerate(handle["metadata/split"]):
            split = decode_h5_string(split_raw)
            if split in selected and len(selected[split]) < args.per_split:
                selected[split].append(index)
        for split, indices in selected.items():
            for index in indices:
                ecg = invert_record(handle, index, restore_baseline=True)
                rate = float(handle["metadata/sampling_rate"][index])
                record_id = decode_h5_string(handle["metadata/record_id"][index])
                patient_id = decode_h5_string(handle["metadata/patient_id"][index])
                time = np.arange(ecg.shape[0]) / rate
                fig, axes = plt.subplots(12, 1, figsize=(14, 14), sharex=True, constrained_layout=True)
                for lead_index, (lead, axis) in enumerate(zip(CANONICAL_LEADS, axes, strict=True)):
                    is_target = lead == target
                    axis.plot(time, ecg[:, lead_index], color="crimson" if is_target else "#315a8a", linewidth=0.7)
                    axis.set_ylabel(lead, rotation=0, ha="right", va="center")
                    if is_target:
                        axis.set_facecolor("#fff0f2")
                        axis.text(0.995, 0.82, "TARGET", transform=axis.transAxes, ha="right", color="crimson")
                    axis.grid(alpha=0.15)
                axes[-1].set_xlabel("Time (s)")
                fig.suptitle(
                    f"CODE-II {split}: patient {patient_id}, record {record_id}, target {target}"
                )
                output = args.output_dir / f"diagnostic_{split}_{safe_name(record_id)}_target_{target}.png"
                fig.savefig(output, dpi=160)
                plt.close(fig)
                print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

