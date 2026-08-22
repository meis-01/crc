from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from physics_esn.pipeline import run_sub001_pipeline


RESERVOIR_MODES = (
    "deterministic_poles",
    "distributed_poles",
    "independent_nonlinear_wc",
    "coupled_nonlinear_wc",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a configuration-driven sub-001 Wilson-Cowan reservoir experiment."
    )
    parser.add_argument("--config", type=Path, default=Path("config") / "defaults.yaml")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--mode",
        choices=RESERVOIR_MODES,
        help="Override reservoir.reservoir_mode for this run.",
    )
    mode_group.add_argument(
        "--all-modes",
        action="store_true",
        help="Run the modes listed under ablation.modes using one shared data split and WC fit.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    if args.all_modes:
        # Keep this import lazy so the original single-mode command remains usable
        # while installations transition to the ablation-capable pipeline.
        from physics_esn.pipeline import run_sub001_ablation

        summary = run_sub001_ablation(args.config, output_dir=args.output_dir)
    elif args.mode is not None:
        summary = run_sub001_pipeline(
            args.config,
            output_dir=args.output_dir,
            reservoir_mode=args.mode,
        )
    else:
        summary = run_sub001_pipeline(args.config, output_dir=args.output_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
