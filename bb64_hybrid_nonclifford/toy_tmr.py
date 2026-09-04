"""Three-qubit local TMR fixture for fast independent simulator checks."""

from __future__ import annotations

from dataclasses import dataclass
import math

from bb64_syndrome_recovery.algebra import angle_table
from bb64_tmr_postselection.circuit import physical_tmr_angle


@dataclass(frozen=True)
class ToyCircuit:
    text: str
    branch_label: int
    postselection_mask: tuple[int, ...]
    expected_probability: float


def build_toy_branch_circuit(*, theta: float, branch_label: int) -> ToyCircuit:
    """Build one local four-branch TMR correction on an X repetition code."""

    if branch_label not in range(4):
        raise ValueError("branch label must lie in 0..3")
    canonical = {
        0: (0, 0, 0),
        1: (1, 0, 0),
        2: (0, 1, 0),
        3: (0, 0, 1),
    }[branch_label]
    syndrome = (canonical[0] ^ canonical[1], canonical[1] ^ canonical[2])
    table = angle_table(theta)
    alpha = physical_tmr_angle(theta, 3)
    lines = [
        "RX 0 1 2",
        "R 3",
        "X 3",
        "M 3",
        f"R_Z({alpha / math.pi:.17g}) 0 1 2",
        "MPP X0*X1",
        "DETECTOR rec[-1]" + (" rec[-2]" if syndrome[0] else ""),
        "MPP X1*X2",
        "DETECTOR rec[-1]" + (" rec[-3]" if syndrome[1] else ""),
    ]
    correction = [str(index) for index, bit in enumerate(canonical) if bit]
    if correction:
        lines.append("Z " + " ".join(correction))
    if branch_label:
        alternative = float(table["alternative"]["logical_angle"])
        lines.append(
            f"R_PAULI({(theta - alternative) / math.pi:.17g}) Z0*Z1*Z2"
        )
    lines.extend(
        [
            f"R_PAULI({-theta / math.pi:.17g}) Z0*Z1*Z2",
            "MPP X0*X1",
            "DETECTOR rec[-1]",
            "MPP X1*X2",
            "DETECTOR rec[-1]",
            "MPP X0",
            "OBSERVABLE_INCLUDE(0) rec[-1]",
        ]
    )
    probability = (
        float(table["probability_target"])
        if branch_label == 0
        else float(table["alternative"]["probability_per_pattern"])
    )
    return ToyCircuit(
        text="\n".join(lines) + "\n",
        branch_label=branch_label,
        postselection_mask=(1, 1, 0, 0),
        expected_probability=probability,
    )
