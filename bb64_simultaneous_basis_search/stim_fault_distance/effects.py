"""Detector-error extraction and physical-translation anchors for BB64."""

from __future__ import annotations

import stim

from n32_k4_d6_reference_code.stim_fault_distance.fault_distance import FaultEffect


def raw_dem_effects(dem: stim.DetectorErrorModel) -> tuple[FaultEffect, ...]:
    """Extract undecomposed DEM errors in their original instruction order."""

    effects = []
    for instruction_index, instruction in enumerate(dem):
        if instruction.type != "error":
            continue
        detector_mask = 0
        observable_mask = 0
        for target in instruction.targets_copy():
            if target.is_separator():
                raise ValueError("unexpected decomposed DEM separator")
            if target.is_relative_detector_id():
                detector_mask ^= 1 << int(target.val)
            elif target.is_logical_observable_id():
                observable_mask ^= 1 << int(target.val)
            else:
                raise ValueError(f"unexpected DEM target: {target}")
        effects.append(
            FaultEffect(
                detector_mask=detector_mask,
                observable_mask=observable_mask,
                probability=float(instruction.args_copy()[0]),
                dem_instruction_index=instruction_index,
            )
        )
    return tuple(effects)


def translation_anchor_variables(
    circuit: stim.Circuit, effects: tuple[FaultEffect, ...]
) -> list[int]:
    """Return logical-crossing faults that can physically occur at cell zero.

    The syndrome circuit has a free 32-element spatial translation action.
    Every nonzero logical witness contains at least one fault whose observable
    component is nonzero. Translating that physical fault to cell zero keeps
    its logical component nonzero. Stim can merge several physical locations
    into one DEM signature, so completeness requires inspecting every circuit
    location associated with each signature instead of just its first one.
    """

    explained = circuit.explain_detector_error_model_errors(
        reduce_to_one_representative_error=False
    )
    if len(explained) != len(effects):
        raise RuntimeError("circuit explanations do not align with DEM fault effects")
    variables = []
    for effect_index, error in enumerate(explained):
        locations = error.circuit_error_locations
        if not locations:
            raise RuntimeError("a DEM effect has no representative circuit location")
        occurs_at_cell_zero = False
        for location in locations:
            qubits = [
                int(target.gate_target.value)
                for target in location.instruction_targets.targets_in_range
                if not target.gate_target.is_combiner
            ]
            data_qubits = [qubit for qubit in qubits if qubit < 64]
            if data_qubits:
                cells = [qubit % 32 for qubit in data_qubits]
            else:
                cells = [
                    qubit - 64 if qubit < 96 else qubit - 96 for qubit in qubits
                ]
            if not cells or any(cell < 0 or cell >= 32 for cell in cells):
                raise RuntimeError(f"could not assign a physical cell to fault {qubits}")
            occurs_at_cell_zero |= 0 in cells
        if occurs_at_cell_zero and effects[effect_index].observable_mask:
            variables.append(effect_index + 1)
    if not variables:
        raise RuntimeError("no logical-crossing faults were found over translation cell zero")
    return variables
