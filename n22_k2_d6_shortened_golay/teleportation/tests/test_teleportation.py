from __future__ import annotations

from itertools import combinations, product
import math
from pathlib import Path
import unittest

import clifft
import numpy as np

from n22_k2_d6_shortened_golay.teleportation.circuit import (
    REFERENCE_QUBITS,
    TeleportationNoise,
    build_estimator_circuit,
)
from n22_k2_d6_shortened_golay.teleportation.decoder import JointPauliDecoder
from n22_k2_d6_shortened_golay.teleportation.rus import (
    InjectionPoint,
    InjectionTable,
    expected_parallel_levels,
    simulate_rus,
)
from n22_k2_d6_shortened_golay.tmr_postselection.circuit import NoiseModel
from n22_k2_d6_shortened_golay.tmr_postselection.model import load_code_artifact
from n22_k2_d6_shortened_golay.tmr_postselection.partitions import (
    certify_partitions,
    load_partitions,
)


PACKAGE_DIR = Path(__file__).resolve().parents[1]


class N22TeleportationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.code = load_code_artifact()
        cls.certificate = certify_partitions(
            cls.code,
            load_partitions(PACKAGE_DIR.parent / "tmr_postselection" / "partitions_m3.json"),
        )

    def test_estimator_circuit_layout_and_counts(self) -> None:
        bundle = build_estimator_circuit(
            code=self.code,
            partition_certificate=self.certificate,
            theta=math.pi / 32,
            logical_count=2,
            preparation_noise=NoiseModel(),
            teleportation_noise=TeleportationNoise(),
        )
        self.assertEqual(REFERENCE_QUBITS, tuple(range(44, 66)))
        self.assertEqual(len(bundle.terminal_z_measurements), 22)
        self.assertEqual(len(bundle.terminal_x_measurements), 22)
        self.assertEqual(sum(bundle.postselection_mask), 22)
        self.assertEqual(bundle.operation_counts["teleportation_cnots"], 22)
        self.assertEqual(bundle.operation_counts["ideal_inverse_logical_rotations"], 2)
        program = clifft.compile(bundle.text, postselection_mask=bundle.postselection_mask)
        self.assertEqual(program.num_qubits, 66)
        self.assertEqual(program.num_observables, 4)

    def test_joint_decoder_corrects_all_weight_two_paulis(self) -> None:
        terminal_z: list[np.ndarray] = []
        terminal_x: list[np.ndarray] = []
        for weight in range(3):
            for support in combinations(range(22), weight):
                for paulis in product((1, 2, 3), repeat=weight):
                    x_error = np.zeros(22, dtype=np.uint8)
                    z_error = np.zeros(22, dtype=np.uint8)
                    for qubit, pauli in zip(support, paulis, strict=True):
                        x_error[qubit] = pauli & 1
                        z_error[qubit] = (pauli >> 1) & 1
                    terminal_z.append(x_error)
                    terminal_x.append(z_error)
        decoder = JointPauliDecoder(self.code, max_search_weight=2)
        decoded = decoder.decode(np.asarray(terminal_z), np.asarray(terminal_x))
        self.assertFalse(np.any(decoded.logical_z))
        self.assertFalse(np.any(decoded.logical_x))
        self.assertLessEqual(int(np.max(decoded.correction_weight)), 2)
        self.assertFalse(np.any(decoded.ambiguous))

    def test_zero_noise_cancel_branch_has_zero_infidelity(self) -> None:
        bundle = build_estimator_circuit(
            code=self.code,
            partition_certificate=self.certificate,
            theta=math.pi / 32,
            logical_count=2,
        )
        program = clifft.compile(bundle.text, postselection_mask=bundle.postselection_mask)
        sample = clifft.sample_survivors(
            program, shots=10_000, seed=2202, keep_records=True
        )
        decoder = JointPauliDecoder(self.code)
        decoded = decoder.decode(
            sample.measurements[:, list(bundle.terminal_z_measurements)],
            sample.measurements[:, list(bundle.terminal_x_measurements)],
        )
        self.assertGreater(sample.passed_shots, 4_000)
        for logical in range(2):
            cancel = decoded.logical_z[:, logical] == 0
            self.assertGreater(int(np.sum(cancel)), 1_800)
            self.assertFalse(np.any(decoded.logical_x[cancel, logical]))
        patterns = decoded.logical_z[:, 0] * 2 + decoded.logical_z[:, 1]
        frequencies = np.bincount(patterns, minlength=4) / len(patterns)
        self.assertTrue(np.all(np.abs(frequencies - 0.25) < 0.04))

    def test_ideal_two_logical_rus_statistics(self) -> None:
        points = []
        for exponent in range(5):
            angle = math.pi / 32 * 2**exponent
            points.extend(
                (
                    InjectionPoint(angle, 1, 3, 1.0, (1.0, 0.0)),
                    InjectionPoint(angle, 2, 3, 1.0, (1.0, 0.0, 0.0, 0.0)),
                )
            )
        result = simulate_rus(
            theta=math.pi / 32,
            logical_count=2,
            shots=100_000,
            table=InjectionTable(points),
            seed=11,
            maximum_levels=8,
        )
        # The dyadic ladder terminates for free when the next correction is pi,
        # so this is just below the uncapped geometric value 8/3.
        capped_theory = sum(1 - (1 - 2.0**-level) ** 2 for level in range(5))
        self.assertAlmostEqual(result["mean_parallel_levels"], capped_theory, delta=0.015)
        self.assertEqual(result["truncated"], 0)
        self.assertEqual(result["any_logical_error_rate"], 0.0)
        self.assertAlmostEqual(expected_parallel_levels(2), 8 / 3, places=12)


if __name__ == "__main__":
    unittest.main()
