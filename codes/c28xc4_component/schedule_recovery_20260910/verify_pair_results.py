#!/usr/bin/env python3
"""Check saved signatures and witnesses; optionally repeat the exhaustive <=4 search."""
import argparse
import hashlib
import json
import struct
import subprocess
from pathlib import Path

import stim

from analyze import ROOT, circuit, effects, save, validate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="rerun exhaustive search (about 2 GB RAM, one basis at a time)")
    args = parser.parse_args()
    certificate = json.loads((ROOT / "pair_results.json").read_text())
    schedule = json.loads((ROOT / certificate["schedule"]).read_text())
    validate(schedule["layers"])
    report = {}
    if args.full:
        subprocess.run(["g++", "-O3", "-std=c++20", str(ROOT / "pair_certify.cpp"), "-o", str(ROOT / "pair_certify")], check=True)
    for basis in ("X", "Z"):
        entry = certificate[basis]
        blob = (ROOT / entry["effects_file"]).read_bytes()
        assert hashlib.sha256(blob).hexdigest() == entry["effects_sha256"]
        n, nd, no, words = struct.unpack_from("<4I", blob)
        assert (n, nd, no, words) == (12908, 288, 16, 6)
        values = []
        for i in range(n):
            chunks = struct.unpack_from("<7Q", blob, 16 + 56*i)
            values.append((sum(v << (64*k) for k, v in enumerate(chunks[:6])), chunks[6]))
        c = circuit(schedule["layers"], basis)
        assert values == effects(c)
        selected, d, o = [], 0, 0
        for i in entry["witness_effect_indices"]:
            di, oi = values[i]
            d ^= di
            o ^= oi
            tokens = [f"D{k}" for k in range(nd) if di >> k & 1] + [f"L{k}" for k in range(no) if oi >> k & 1]
            explained = c.explain_detector_error_model_errors(
                dem_filter=stim.DetectorErrorModel("error(1) " + " ".join(tokens)),
                reduce_to_one_representative_error=True,
            )
            assert explained and explained[0].circuit_error_locations
            selected.append({"effect_index": i, "effect": " ".join(tokens), "explanation": str(explained[0])})
        assert d == 0 and o != 0
        record = {"signature_reconstruction_verified": True, "four_fault_witness_verified": True,
                  "combined_observables": [k for k in range(no) if o >> k & 1], "witness": selected}
        if args.full:
            run = subprocess.run([str(ROOT / "pair_certify"), str(ROOT / entry["effects_file"])], capture_output=True, text=True)
            result = json.loads(run.stdout.strip().splitlines()[-1])
            assert run.returncode == 1 and result["faults"] == 4
            record["exhaustive_rerun"] = result
            record["exhaustive_log"] = run.stderr
        report[basis] = record
        print(json.dumps({"basis": basis, **{k:v for k,v in record.items() if k != "witness"}}), flush=True)
    save(ROOT / "pair_witness_validation.json", report)


if __name__ == "__main__":
    main()
