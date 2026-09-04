"""Tests for exact scheduled-circuit signatures and action posteriors."""

from __future__ import annotations

import math
import unittest

import numpy as np

from bb64_hybrid_nonclifford.circuit import class_id_for_labels
from bb64_hybrid_nonclifford.circuit_decoder.branches import (
    branch_action_data,
    rotation_layout,
    signed_local_branch,
)
from bb64_hybrid_nonclifford.circuit_decoder.belief_propagation import (
    CategoricalBPDecoder,
)
from bb64_hybrid_nonclifford.circuit_decoder.catalog import (
    FaultCatalog,
    SignatureGroup,
    build_fault_catalog,
)
from bb64_hybrid_nonclifford.circuit_decoder.decoder import ScheduledActionDecoder
from bb64_hybrid_nonclifford.circuit_decoder.factor_graph import (
    D3FactorGraph,
    _branch_variable,
    _component_data,
    _physical_variable,
)
from bb64_hybrid_nonclifford.circuit_decoder.frames import (
    BoundaryFrame,
    GF2RowReducer,
    bits_to_int,
)
from bb64_hybrid_nonclifford.circuit_decoder.ledger import FaultMechanism, build_fault_ledger
from bb64_hybrid_nonclifford.circuit_decoder.list_decoder import D3ListDecoder
from bb64_hybrid_nonclifford.circuit_decoder.labeled_noise import labeled_trajectory
from bb64_hybrid_nonclifford.circuit_decoder.propagate import (
    PauliSignature,
    parse_circuit,
    propagate_faults,
)
from bb64_hybrid_nonclifford.model import load_hybrid_model
from bb64_hybrid_nonclifford.syndrome_history import build_syndrome_history_circuit
from bb64_syndrome_recovery.algebra import angle_table
from bb64_tmr_postselection.circuit import NoiseModel


class ScheduledDecoderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = load_hybrid_model()
        cls.theta = math.pi / 32
        cls.history = build_syndrome_history_circuit(
            cls.model,
            theta=cls.theta,
            noise=NoiseModel(probability=0.001),
            syndrome_rounds=3,
        )
        cls.parsed = parse_circuit(cls.history.text)
        cls.ledger = build_fault_ledger(cls.history.text)

    def test_frozen_ledger_size_and_probability_partition(self) -> None:
        self.assertEqual(self.parsed.num_qubits, 128)
        self.assertEqual(self.parsed.num_measurements, 224)
        self.assertEqual(self.parsed.num_detectors, 196)
        self.assertEqual(self.parsed.num_rotations, 24)
        self.assertEqual(len(self.ledger.locations), 3016)
        self.assertEqual(len(self.ledger.mechanisms), 31064)
        self.assertAlmostEqual(
            self.ledger.through_two_fault_probability + self.ledger.higher_fault_tail_probability,
            1.0,
            places=14,
        )

    def test_empty_and_pair_propagation_are_linear(self) -> None:
        self.assertEqual(propagate_faults(self.parsed, ()), PauliSignature(0, 0, 0, 0))
        left = self.ledger.mechanisms[2000]
        right = next(
            mechanism
            for mechanism in self.ledger.mechanisms[4000:]
            if mechanism.location_id != left.location_id
        )
        a = propagate_faults(self.parsed, (left,))
        b = propagate_faults(self.parsed, (right,))
        pair = propagate_faults(self.parsed, (left, right))
        self.assertEqual(pair.detector_mask, a.detector_mask ^ b.detector_mask)
        self.assertEqual(pair.final_x_mask, a.final_x_mask ^ b.final_x_mask)
        self.assertEqual(pair.final_z_mask, a.final_z_mask ^ b.final_z_mask)
        self.assertEqual(pair.rotation_sign_mask, a.rotation_sign_mask ^ b.rotation_sign_mask)

    def test_fault_crossing_rotation_flips_only_the_before_sign(self) -> None:
        rotation = next(item for item in self.parsed.instructions if item.name == "R_Z")
        pivot = int(rotation.targets[0])
        common = dict(
            mechanism_id=-1,
            location_id=-1,
            channel="FORCED",
            targets=(pivot,),
            pauli="X",
            x_mask=1 << pivot,
            z_mask=0,
            probability=1.0,
            odds=1.0,
        )
        before = FaultMechanism(instruction_index=rotation.index - 1, **common)
        after = FaultMechanism(instruction_index=rotation.index, **common)
        self.assertEqual(propagate_faults(self.parsed, (before,)).rotation_sign_mask & 1, 1)
        self.assertEqual(propagate_faults(self.parsed, (after,)).rotation_sign_mask & 1, 0)

    def test_signed_branch_reduces_to_existing_angle_table(self) -> None:
        table = angle_table(self.theta)
        target = signed_local_branch(self.theta, np.zeros(3, dtype=np.uint8), np.ones(3, dtype=np.int8))
        alternative = signed_local_branch(
            self.theta,
            np.asarray((1, 0, 0), dtype=np.uint8),
            np.ones(3, dtype=np.int8),
        )
        self.assertAlmostEqual(target[0], table["probability_target"], places=14)
        self.assertAlmostEqual(target[1], table["target"]["logical_angle"], places=14)
        self.assertAlmostEqual(
            alternative[0], table["alternative"]["probability_per_pattern"], places=14
        )
        self.assertAlmostEqual(
            alternative[1], table["alternative"]["logical_angle"], places=14
        )

    def test_rotation_layout_matches_generated_targets(self) -> None:
        expected = tuple(pivot for _logical, _piece, pivot in rotation_layout(self.model))
        actual = tuple(
            int(target)
            for instruction in self.parsed.instructions
            if instruction.name == "R_Z"
            for target in instruction.targets
        )
        self.assertEqual(actual, expected)

    def test_no_fault_action_decodes_an_ideal_branch(self) -> None:
        mini = build_fault_catalog(
            self.history,
            self.model.code,
            mechanism_ids=(),
        )
        decoder = ScheduledActionDecoder(
            self.model,
            mini,
            theta=self.theta,
            minimum_posterior=0.0,
        )
        class_id = class_id_for_labels(self.model, (1, 0, 2, 0, 0, 0, 0, 0))
        syndrome = self.model.displayed_syndromes[class_id]
        observed = 0
        for group in self.history.x_detector_groups:
            for check, detector in enumerate(group):
                observed |= int(syndrome[check]) << detector
        result = decoder.decode_mask(observed)
        self.assertFalse(result.reset)
        self.assertIsNotNone(result.action)
        assert result.action is not None
        action_mask = sum(1 << qubit for qubit in result.action.physical_z_correction)
        original_mask = bits_to_int(
            self.model.recovery.correction_physical_supports[class_id]
        )
        self.assertEqual(
            GF2RowReducer(self.model.code.matrix_z).reduce(action_mask ^ original_mask),
            0,
        )

    def test_equivalent_explanations_are_summed_by_action(self) -> None:
        empty = PauliSignature(0, 0, 0, 0)
        zero = BoundaryFrame(0, 0, 0, 0)
        groups = (
            SignatureGroup(empty, zero, 0.1, 1, ((0, 0.1),), 0),
            SignatureGroup(empty, zero, 0.2, 1, ((1, 0.2),), 1),
        )
        mini = FaultCatalog(
            self.parsed,
            self.history,
            self.ledger,
            SignatureGroup(empty, zero, 1.0, 0, (), -1),
            groups,
        )
        decoder = ScheduledActionDecoder(
            self.model, mini, theta=self.theta, minimum_posterior=0.0
        )
        result = decoder.decode_mask(0)
        self.assertEqual(result.matching_explanations, 3)
        self.assertEqual(result.distinct_actions, 1)
        self.assertAlmostEqual(result.modeled_posterior, 1.0)

    def test_class_action_probability_is_normalized_locally(self) -> None:
        probabilities = 0.0
        for label in range(4):
            class_id = class_id_for_labels(self.model, (label, 0, 0, 0, 0, 0, 0, 0))
            data = branch_action_data(
                self.model,
                theta=self.theta,
                class_id=class_id,
                rotation_sign_mask=0,
                logical_x_frame=0,
            )
            # Divide out the seven all-target logical factors.
            probabilities += data.probability / angle_table(self.theta)["probability_target"] ** 7
        self.assertAlmostEqual(probabilities, 1.0, places=13)

    def test_row_reducer_canonicalizes_stabilizer_cosets(self) -> None:
        reducer = GF2RowReducer(self.model.code.matrix_z)
        for row in self.model.code.matrix_z[:8]:
            row_mask = bits_to_int(row)
            self.assertEqual(reducer.reduce(row_mask), 0)
            value = (1 << 1) | (1 << 17) | (1 << 53)
            self.assertEqual(reducer.reduce(value), reducer.reduce(value ^ row_mask))

    def test_branch_only_d3_graph_decodes_every_local_label(self) -> None:
        variables = tuple(
            _branch_variable(self.model, self.history, logical, self.theta)
            for logical in range(8)
        )
        offsets, detectors = _component_data(variables)
        graph = D3FactorGraph(
            variables=variables,
            physical_variable_count=0,
            parsed_circuit=self.parsed,
            history=self.history,
            ledger=self.ledger,
            component_offsets=offsets,
            component_detector_masks=detectors,
        )
        labels = (0, 1, 2, 3, 0, 3, 2, 1)
        codes = tuple(
            variable.state_semantics.index(label)
            for variable, label in zip(variables, labels, strict=True)
        )
        observed = graph.detector_mask(codes)
        bp = CategoricalBPDecoder(graph).decode(observed)
        self.assertEqual(bp.residual_detector_mask, 0)
        decoder = D3ListDecoder(
            graph,
            self.model,
            theta=self.theta,
            osd_order=0,
            osd_window=0,
            maximum_candidates=1,
            minimum_action_probability=0.0,
        )
        result = decoder.decode(observed)
        expected, _probability = decoder.action_builder.from_signature(
            class_id_for_labels(self.model, labels),
            final_x_mask=0,
            final_z_mask=0,
            rotation_sign_mask=0,
        )
        self.assertFalse(result.reset)
        self.assertEqual(result.action, expected)

    def test_categorical_locations_preserve_native_pauli_channels(self) -> None:
        expected_sizes = {
            "DEPOLARIZE1": 4,
            "DEPOLARIZE2": 16,
            "X_ERROR": 2,
            "Z_ERROR": 2,
        }
        for kind, size in expected_sizes.items():
            location = next(item for item in self.ledger.locations if item.channel == kind)
            variable = _physical_variable(self.parsed, self.ledger, location)
            self.assertEqual(len(variable.state_codes), size)
            self.assertAlmostEqual(
                sum(math.exp(value) for value in variable.log_priors), 1.0, places=14
            )
            for code, signature in enumerate(variable.signatures):
                predicted = PauliSignature(0, 0, 0, 0)
                for bit in range(variable.bit_width):
                    if (code >> bit) & 1:
                        basis = variable.signatures[1 << bit]
                        predicted = PauliSignature(
                            predicted.detector_mask ^ basis.detector_mask,
                            predicted.final_x_mask ^ basis.final_x_mask,
                            predicted.final_z_mask ^ basis.final_z_mask,
                            predicted.rotation_sign_mask ^ basis.rotation_sign_mask,
                        )
                self.assertEqual(signature, predicted)

    def test_labeled_multifault_trajectory_matches_direct_propagation(self) -> None:
        locations = tuple(
            next(item for item in self.ledger.locations if item.channel == kind)
            for kind in ("DEPOLARIZE1", "DEPOLARIZE2", "X_ERROR", "Z_ERROR")
        )
        physical = tuple(
            _physical_variable(self.parsed, self.ledger, location)
            for location in locations
        )
        branches = tuple(
            _branch_variable(self.model, self.history, logical, self.theta)
            for logical in range(8)
        )
        variables = physical + branches
        offsets, detectors = _component_data(variables)
        graph = D3FactorGraph(
            variables=variables,
            physical_variable_count=len(physical),
            parsed_circuit=self.parsed,
            history=self.history,
            ledger=self.ledger,
            component_offsets=offsets,
            component_detector_masks=detectors,
        )
        branch_labels = (3, 0, 1, 0, 2, 0, 0, 1)
        codes = tuple(1 for _ in physical) + tuple(
            variable.state_semantics.index(label)
            for variable, label in zip(branches, branch_labels, strict=True)
        )
        from bb64_hybrid_nonclifford.circuit_decoder.action_model import RepairActionBuilder

        trajectory = labeled_trajectory(
            graph,
            self.model,
            RepairActionBuilder(self.model, theta=self.theta),
            codes,
        )
        mechanisms = tuple(
            self.ledger.mechanisms[variable.state_semantics[1]] for variable in physical
        )
        direct = propagate_faults(self.parsed, mechanisms)
        self.assertEqual(trajectory.fault_count, 4)
        self.assertEqual(trajectory.physical_signature, direct)
        branch_detector = graph.detector_mask((0,) * len(physical) + codes[len(physical) :])
        self.assertEqual(
            trajectory.observed_detector_mask,
            direct.detector_mask ^ branch_detector,
        )


if __name__ == "__main__":
    unittest.main()
