"""Create compact CSV/Markdown reports from completed paired-policy runs."""

import argparse
import csv
import json
import math
from pathlib import Path


def summarize(input_dir, output_dir):
    records = [json.loads(p.read_text()) for p in sorted(input_dir.glob("*_N*.json"))]
    if not records or any(not r["completed"] for r in records):
        raise ValueError("summary requires nonempty, completed fixed-shot runs")
    output_dir.mkdir(parents=True, exist_ok=False)
    controls = {(r["configuration"]["presentation"], r["configuration"]["logical_count"]): r
                for r in records if r["configuration"]["theta"] == 0}
    rows = []
    for r in records:
        c, counts, circuit = r["configuration"], r["counts"], r["circuit"]
        control = controls.get((c["presentation"], c["logical_count"]))
        for policy in ("strict-xz", "tmr-x"):
            est = r["estimates"][policy]
            base = control["estimates"][policy] if control else None
            ratio = est["acceptance_over_ideal"]/base["acceptance"] if base and base["acceptance"] else None
            rows.append({"presentation": c["presentation"], "N": c["logical_count"],
                         "theta": c["theta"], "angle_label": c["angle_label"], "p": c["probability"],
                         "M": c["partition_count"], "policy": policy, "shots": counts["attempted"],
                         "accepted": est["accepted"], "acceptance": est["acceptance"],
                         "wilson_low": est["wilson_95"][0], "wilson_high": est["wilson_95"][1],
                         "standard_error": est["standard_error"],
                         "ideal_acceptance": r["ideal_acceptance"],
                         "noise_survival": est["acceptance_over_ideal"],
                         "noise_survival_over_zero_angle": ratio,
                         "resources_per_attempt": est["resources_per_attempt"],
                         "attempts_per_block": est["attempts_per_accepted_block"],
                         "cnots_per_resource": est["cnots_per_candidate_resource"],
                         "cnot_layers_per_resource": est["cnot_layers_per_candidate_resource"],
                         "qubit_cnot_layers_per_resource": est["allocated_qubit_cnot_layers_per_candidate_resource"],
                         "physical_cnots": circuit["operation_counts"]["physical_cnots"],
                         "cnot_depth": circuit["operation_counts"]["cnot_layers"],
                         "qubits": circuit["qubits"], "peak_active_width": circuit["peak_active_width"],
                         "sampling_seconds": counts["sampling_seconds"]})
    with (output_dir/"acceptance.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# 313c9898 one-/four-resource post-selection results", "",
             "Both policies use identical trajectories at each point. Intervals are pointwise 95% Wilson intervals.",
             "Accepted resources are candidates; conditional noisy logical fidelity has not been measured.", "",
             "| Schedule | N | Angle | Strict XZ acceptance (95% CI) | X-only acceptance (95% CI) | Ideal |",
             "|---|---:|---|---:|---:|---:|"]
    for r in records:
        c = r["configuration"]
        def percent(policy):
            e = r["estimates"][policy]
            a, b = e["wilson_95"]
            return f"{100*e['acceptance']:.3f}% [{100*a:.3f}, {100*b:.3f}]"
        lines.append(f"| `{c['presentation']}` | {c['logical_count']} | `{c['angle_label']}` | "
                     f"{percent('strict-xz')} | {percent('tmr-x')} | {100*r['ideal_acceptance']:.3f}% |")
    lines += ["", "## Cost at the pi/32 benchmark", "",
              "| Schedule | N | Policy | Resources/attempt | Attempts/block | CNOTs/resource | Allocated qubit-CNOT-layers/resource |",
              "|---|---:|---|---:|---:|---:|---:|"]
    for row in rows:
        if row["angle_label"] != "pi32":
            continue
        lines.append(f"| `{row['presentation']}` | {row['N']} | `{row['policy']}` | "
                     f"{row['resources_per_attempt']:.5f} | {row['attempts_per_block']:.3f} | "
                     f"{row['cnots_per_resource']:.1f} | {row['qubit_cnot_layers_per_resource']:.1f} |")
    lines += ["", "The CSV also contains zero-angle-normalized survival ratios. These test an empirical",
              "angle-independent noise-survival approximation; BB56 fit constants are not reused.", "",
              "Costs include one Z-only initialization round, the rotation gadgets, and one full",
              "terminal X/Z round. There is no physical early-abort saving in this protocol.",
              "CNOT depth excludes reset, measurement and the single rotation layer. Qubit-layer",
              "cost is an allocation proxy, not hardware-calibrated elapsed time.", "",
              f"Total sampled attempts in this report: {sum(r['counts']['attempted'] for r in records):,}.",
              f"Total sampling time: {sum(r['counts']['sampling_seconds'] for r in records):.3f} seconds.", ""]
    (output_dir/"searches/paired_polynomial/RESULTS.md").write_text("\n".join(lines))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = summarize(args.input, args.output)
    print(f"wrote {len(rows)} policy rows to {args.output}")


if __name__ == "__main__":
    main()
