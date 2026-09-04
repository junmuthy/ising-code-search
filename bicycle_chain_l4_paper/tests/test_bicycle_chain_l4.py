from __future__ import annotations

import math
from pathlib import Path
import unittest

import clifft
import numpy as np
import stim

from bicycle_chain_l2_paper.tmr_postselection.circuit import (
    NoiseModel,
    build_circuit,
    encoded_plus_circuit,
    ideal_tmr_acceptance,
)
from bicycle_chain_l2_paper.tmr_postselection.model import initialization_correction_map
from bicycle_chain_l2_paper.tmr_postselection.partitions import (
    certify_partitions,
    load_partitions,
)
from bicycle_chain_l4_paper.code_data.code import validate_code
from bicycle_chain_l4_paper.tmr_postselection.model import (
    EXPECTED_BASIS_SHA256,
    EXPECTED_SCHEDULE_SHA256,
    load_code_artifact,
)
from bicycle_chain_l4_paper.tmr_postselection.analyze_suite import (
    expected_geometric_maximum,
    geometric_maximum_quantile,
)


PACKAGE_DIR = Path(__file__).resolve().parents[1]


class BicycleChainL4Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.code = load_code_artifact()
        cls.certificate = certify_partitions(
            cls.code,
            load_partitions(PACKAGE_DIR / "tmr_postselection" / "partitions_m3.json"),
        )

    def test_static_structure(self) -> None:
        result = validate_code(exhaustive=False)
        self.assertEqual(result["parameters"], "[[56,8,6]]")
        self.assertEqual((result["rank_x"], result["rank_z"]), (24, 24))
        self.assertEqual(
            result["translation_x_logical_action"], (1, 2, 3, 0, 5, 6, 7, 4)
        )

    def test_frozen_artifacts_and_paper_schedule(self) -> None:
        self.assertEqual(self.code.basis_sha256, EXPECTED_BASIS_SHA256)
        self.assertEqual(self.code.schedule_sha256, EXPECTED_SCHEDULE_SHA256)
        self.assertEqual(len(self.code.schedule), 8)
        self.assertEqual(sum(map(len, self.code.schedule)), 448)
        for layer in self.code.schedule:
            self.assertEqual(len(layer), 56)
            self.assertEqual(len({gate.data for gate in layer}), 56)

    def test_disjoint_logicals_and_tmr_kernel(self) -> None:
        supports = [set(support) for support in self.code.logicals_z]
        self.assertEqual(sum(map(len, supports)), 56)
        self.assertEqual(len(set.union(*supports)), 56)
        self.assertEqual(self.certificate.batch_ranks, (16,))
        self.assertEqual(self.certificate.combined_rank, 16)
        self.assertEqual(self.certificate.combined_kernel_dimension, 8)
        self.assertTrue(self.certificate.single_final_check_certified)

    def test_initialization_right_inverse(self) -> None:
        rows, correction_qubits, right_inverse = initialization_correction_map(
            self.code.matrix_z
        )
        self.assertEqual(len(rows), 24)
        self.assertEqual(len(correction_qubits), 24)
        self.assertTrue(
            np.array_equal(
                (self.code.matrix_z[list(rows)] @ right_inverse) % 2,
                np.eye(24, dtype=np.uint8),
            )
        )

    def test_exact_encoder(self) -> None:
        simulator = stim.TableauSimulator()
        simulator.do_circuit(encoded_plus_circuit(self.code))
        for axis, supports in (("X", self.code.checks_x), ("Z", self.code.checks_z)):
            for support in supports:
                pauli = stim.PauliString(56)
                for qubit in support:
                    pauli[qubit] = axis
                self.assertEqual(simulator.peek_observable_expectation(pauli), 1)

    def test_n1_operation_counts_and_noiseless_limit(self) -> None:
        bundle = build_circuit(
            code=self.code,
            partition_certificate=self.certificate,
            mode="full",
            protocol="single-final-check",
            theta=math.pi / 32,
            logical_count=1,
            noise=NoiseModel(),
        )
        self.assertEqual(bundle.operation_counts["physical_cnots"], 680)
        self.assertEqual(bundle.operation_counts["cnot_layers"], 20)
        self.assertEqual(bundle.operation_counts["physical_rotations"], 3)
        self.assertEqual(sum(bundle.postselection_mask), 56)
        self.assertAlmostEqual(
            bundle.ideal_acceptance, ideal_tmr_acceptance(math.pi / 32, 3)
        )
        zero = build_circuit(
            code=self.code,
            partition_certificate=self.certificate,
            mode="full",
            protocol="single-final-check",
            theta=0,
            logical_count=1,
            noise=NoiseModel(),
        )
        program = clifft.compile(zero.text, postselection_mask=zero.postselection_mask)
        sample = clifft.sample_survivors(program, shots=16, seed=17)
        self.assertEqual(sample.passed_shots, 16)
        self.assertEqual(program.num_qubits, 112)
        all_logicals = build_circuit(
            code=self.code,
            partition_certificate=self.certificate,
            mode="full",
            protocol="single-final-check",
            theta=math.pi / 32,
            logical_count=8,
            noise=NoiseModel(probability=1e-3),
        )
        self.assertEqual(all_logicals.operation_counts["physical_cnots"], 736)
        self.assertEqual(all_logicals.operation_counts["cnot_layers"], 20)
        self.assertEqual(all_logicals.operation_counts["physical_rotations"], 24)

    def test_geometric_block_pool_statistics(self) -> None:
        self.assertAlmostEqual(expected_geometric_maximum(0.5, 1), 2.0)
        self.assertAlmostEqual(expected_geometric_maximum(1.0, 16), 1.0)
        self.assertEqual(geometric_maximum_quantile(0.5, 1, 0.5), 1)
        self.assertEqual(geometric_maximum_quantile(0.5, 2, 0.5), 2)


if __name__ == "__main__":
    unittest.main()
