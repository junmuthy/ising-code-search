from __future__ import annotations

import math
from pathlib import Path
import unittest

import clifft
import numpy as np
import stim

from bb64_tmr_postselection.circuit import (
    NoiseModel,
    build_circuit,
    encoded_plus_circuit,
    ideal_tmr_acceptance,
    physical_tmr_angle,
)
from bb64_tmr_postselection.model import (
    BATCHES,
    EXPECTED_BASIS_SHA256,
    EXPECTED_SCHEDULE_SHA256,
    initialization_correction_map,
    load_code_artifact,
)
from bb64_tmr_postselection.partitions import (
    certify_partitions,
    load_partitions,
)


PACKAGE_DIR = Path(__file__).resolve().parents[1]


class BB64PostselectionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.code = load_code_artifact()
        cls.certificate = certify_partitions(
            cls.code, load_partitions(PACKAGE_DIR / "partitions_m3.json")
        )

    def test_frozen_artifacts_and_batches(self) -> None:
        self.assertEqual(self.code.basis_sha256, EXPECTED_BASIS_SHA256)
        self.assertEqual(self.code.schedule_sha256, EXPECTED_SCHEDULE_SHA256)
        self.assertEqual(self.code.matrix_x.shape, (32, 64))
        for batch in BATCHES:
            supports = [set(self.code.logicals_z[index]) for index in batch]
            self.assertEqual(sum(map(len, supports)), 32)
            self.assertEqual(len(set.union(*supports)), 32)

    def test_partition_kernel_certificate(self) -> None:
        self.assertEqual(self.certificate.batch_ranks, (8, 8))
        self.assertEqual(self.certificate.batch_kernel_dimensions, (4, 4))
        self.assertEqual(self.certificate.combined_rank, 16)
        self.assertEqual(self.certificate.combined_kernel_dimension, 8)
        self.assertTrue(self.certificate.single_final_check_certified)

    def test_initialization_right_inverse(self) -> None:
        rows, correction_qubits, right_inverse = initialization_correction_map(
            self.code.matrix_z
        )
        self.assertEqual(len(rows), 28)
        self.assertEqual(len(correction_qubits), 28)
        self.assertTrue(
            np.array_equal(
                (self.code.matrix_z[list(rows)] @ right_inverse) % 2,
                np.eye(28, dtype=np.uint8),
            )
        )

    def test_exact_encoder_prepares_checks_and_logical_plus(self) -> None:
        simulator = stim.TableauSimulator()
        simulator.do_circuit(encoded_plus_circuit(self.code))
        for axis, supports in (("X", self.code.checks_x), ("Z", self.code.checks_z)):
            for support in supports:
                pauli = stim.PauliString(64)
                for qubit in support:
                    pauli[qubit] = axis
                self.assertEqual(simulator.peek_observable_expectation(pauli), 1)
        for support in self.code.logicals_x:
            pauli = stim.PauliString(64)
            for qubit in support:
                pauli[qubit] = "X"
            self.assertEqual(simulator.peek_observable_expectation(pauli), 1)

    def test_tmr_angle_and_acceptance(self) -> None:
        theta = math.pi / 8
        theta_star = physical_tmr_angle(theta, 3)
        self.assertLess(theta_star, 0)
        direct = math.cos(theta_star / 2) ** 6 + math.sin(theta_star / 2) ** 6
        self.assertAlmostEqual(ideal_tmr_acceptance(theta, 3), direct, places=14)
        self.assertEqual(ideal_tmr_acceptance(theta, 1), 1.0)

    def test_operation_counts(self) -> None:
        single = build_circuit(
            code=self.code,
            partition_certificate=self.certificate,
            mode="full",
            protocol="single-final-check",
            theta=math.pi / 32,
            logical_count=8,
            noise=NoiseModel(),
        )
        self.assertEqual(single.operation_counts["physical_cnots"], 848)
        self.assertEqual(single.operation_counts["cnot_layers"], 24)
        self.assertEqual(sum(single.postselection_mask), 64)
        self.assertEqual(len(single.postselection_mask), 68)

        double = build_circuit(
            code=self.code,
            partition_certificate=self.certificate,
            mode="full",
            protocol="two-projection",
            theta=math.pi / 32,
            logical_count=8,
            noise=NoiseModel(),
        )
        self.assertEqual(double.operation_counts["physical_cnots"], 1360)
        self.assertEqual(double.operation_counts["cnot_layers"], 32)
        self.assertEqual(sum(double.postselection_mask), 128)
        self.assertEqual(len(double.postselection_mask), 132)

    def test_zero_angle_full_circuit_always_survives(self) -> None:
        bundle = build_circuit(
            code=self.code,
            partition_certificate=self.certificate,
            mode="full",
            protocol="single-final-check",
            theta=0,
            logical_count=8,
            noise=NoiseModel(),
        )
        program = clifft.compile(bundle.text, postselection_mask=bundle.postselection_mask)
        sample = clifft.sample_survivors(program, shots=32, seed=17)
        self.assertEqual(sample.passed_shots, 32)
        self.assertEqual(program.num_qubits, 128)


if __name__ == "__main__":
    unittest.main()
