#!/usr/bin/env python3
"""Run a resumable ``[[56,8,6]]`` bicycle-chain post-selection experiment."""

from pathlib import Path

from bicycle_chain_l2_paper.tmr_postselection.run_postselection import main

from .model import load_code_artifact


DEFAULT_PARTITIONS = Path(__file__).resolve().parent / "partitions_m3.json"


if __name__ == "__main__":
    main(
        code_loader=load_code_artifact,
        default_partitions=DEFAULT_PARTITIONS,
        max_logicals=8,
        default_postselection_policy="tmr-x",
    )
