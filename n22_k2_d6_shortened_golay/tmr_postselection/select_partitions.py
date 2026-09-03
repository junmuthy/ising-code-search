#!/usr/bin/env python3
"""Write the balanced ``[[22,2,6]]`` TMR partition certificate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .model import load_code_artifact
from .partitions import (
    certify_partitions,
    deterministic_balanced_partitions,
    save_certificate,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "partitions_m3.json",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise SystemExit(f"refusing to overwrite {args.output}")
    code = load_code_artifact()
    certificate = certify_partitions(code, deterministic_balanced_partitions(code))
    save_certificate(args.output, certificate)
    print(json.dumps(certificate.to_json(), indent=2), flush=True)
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
