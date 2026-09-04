from __future__ import annotations

import math
from pathlib import Path
import unittest

import clifft
import numpy as np
import stim

from bicycle_chain_l2_paper.code_data.code import validate_code
from bicycle_chain_l2_paper.stim_fault_distance.circuit import (
    build_memory_circuit,
)
from bicycle_chain_l2_paper.stim_fault_distance.model import load_code_data
from bicycle_chain_l2_paper.tmr_postselection.circuit import (
    NoiseModel,
    build_circuit,
    encoded_plus_circuit,
    ideal_tmr_acceptance,
    physical_tmr_angle,
)
from bicycle_chain_l2_paper.tmr_postselection.model import (
    BATCHES,
    EXPECTED_BASIS_SHA256,
    EXPECTED_SCHEDULE_SHA256,
    initialization_correction_map,
    load_code_artifact,
)
from bicycle_chain_l2_paper.tmr_postselection.partitions import (
    certify_partitions,
    load_partitions,
)


PACKAGE_DIR = Path(__file__).resolve().parents[1]


class BicycleChainL2Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.schedule_code = load_code_data()
        cls.code = load_code_artifact()
        cls.certificate = certify_partitions(
            cls.code, load_partitions(PACKAGE_DIR / "tmr_postselection" / "partitions_m3.json")
        )

    def test_exact_static_code(self) -> None:
        result = validate_code(exhaustive=True)
        self.assertEqual(result["parameters"], "[[28,4,5]]")
        self.assertEqual(result["distance_x"], 5)
        self.assertEqual(result["distance_z"], 5)
        self.assertEqual(result["translation_x_logical_action"], (1, 0, 3, 2))
        self.assertEqual(result["redundant_relations"], ((0, 1, 2, 3, 4, 5, 6), (7, 8, 9, 10, 11, 12, 13)))

    def test_paper_schedule(self) -> None:
        self.assertEqual(len(self.schedule_code.schedule), 8)
        self.assertEqual(sum(map(len, self.schedule_code.schedule)), 224)
        self.assertEqual(len(self.schedule_code.redundant_relations), 2)
        for layer in self.schedule_code.schedule:
            self.assertEqual(len(layer), 28)
            self.assertEqual(len({gate.data for gate in layer}), 28)

    def test_noiseless_memory(self) -> None:
        for basis in ("X", "Z"):
            circuit = build_memory_circuit(basis, 3, code=self.schedule_code)
            detectors, observables = circuit.compile_detector_sampler().sample(
                shots=16, separate_observables=True
            )
            self.assertFalse(np.any(detectors))
            self.assertFalse(np.any(observables))
            self.assertEqual(circuit.num_observables, 4)

    def test_frozen_artifacts_and_disjoint_logicals(self) -> None:
        self.assertEqual(self.code.basis_sha256, EXPECTED_BASIS_SHA256)
        self.assertEqual(self.code.schedule_sha256, EXPECTED_SCHEDULE_SHA256)
        self.assertEqual(self.code.matrix_x.shape, (14, 28))
        supports = [set(self.code.logicals_z[index]) for index in BATCHES[0]]
        self.assertEqual(sum(map(len, supports)), 28)
        self.assertEqual(len(set.union(*supports)), 28)

    def test_partition_kernel_certificate(self) -> None:
        self.assertEqual(self.certificate.batch_ranks, (8,))
        self.assertEqual(self.certificate.batch_kernel_dimensions, (4,))
        self.assertEqual(self.certificate.combined_rank, 8)
        self.assertEqual(self.certificate.combined_kernel_dimension, 4)
        self.assertTrue(self.certificate.single_final_check_certified)

    def test_initialization_right_inverse(self) -> None:
        rows, correction_qubits, right_inverse = initialization_correction_map(
            self.code.matrix_z
        )
        self.assertEqual(len(rows), 12)
        self.assertEqual(len(correction_qubits), 12)
        self.assertTrue(
            np.array_equal(
                (self.code.matrix_z[list(rows)] @ right_inverse) % 2,
                np.eye(12, dtype=np.uint8),
            )
        )

    def test_exact_encoder(self) -> None:
        simulator = stim.TableauSimulator()
        simulator.do_circuit(encoded_plus_circuit(self.code))
        for axis, supports in (("X", self.code.checks_x), ("Z", self.code.checks_z)):
            for support in supports:
                pauli = stim.PauliString(28)
                for qubit in support:
                    pauli[qubit] = axis
                self.assertEqual(simulator.peek_observable_expectation(pauli), 1)
        for support in self.code.logicals_x:
            pauli = stim.PauliString(28)
            for qubit in support:
                pauli[qubit] = "X"
            self.assertEqual(simulator.peek_observable_expectation(pauli), 1)

    def test_tmr_angle_and_operation_counts(self) -> None:
        theta = math.pi / 32
        theta_star = physical_tmr_angle(theta, 3)
        direct = math.cos(theta_star / 2) ** 6 + math.sin(theta_star / 2) ** 6
        self.assertAlmostEqual(ideal_tmr_acceptance(theta, 3), direct, places=14)
        bundle = build_circuit(
            code=self.code,
            partition_certificate=self.certificate,
            mode="full",
            protocol="single-final-check",
            theta=theta,
            logical_count=4,
            noise=NoiseModel(),
        )
        self.assertEqual(bundle.operation_counts["physical_cnots"], 368)
        self.assertEqual(bundle.operation_counts["cnot_layers"], 20)
        self.assertEqual(bundle.operation_counts["physical_rotations"], 12)
        self.assertEqual(sum(bundle.postselection_mask), 28)
        self.assertEqual(len(bundle.postselection_mask), 30)
        self.assertAlmostEqual(bundle.ideal_acceptance, 0.22294876010395345)

    def test_zero_angle_full_circuit_survives(self) -> None:
        bundle = build_circuit(
            code=self.code,
            partition_certificate=self.certificate,
            mode="full",
            protocol="single-final-check",
            theta=0,
            logical_count=4,
            noise=NoiseModel(),
        )
        program = clifft.compile(bundle.text, postselection_mask=bundle.postselection_mask)
        sample = clifft.sample_survivors(program, shots=32, seed=17)
        self.assertEqual(sample.passed_shots, 32)
        self.assertEqual(program.num_qubits, 56)


if __name__ == "__main__":
    unittest.main()
