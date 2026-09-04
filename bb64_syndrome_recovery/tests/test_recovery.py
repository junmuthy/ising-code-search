"""Regression tests for exact BB64 syndrome recovery."""

from __future__ import annotations

import math
import unittest

import numpy as np

from bb64_syndrome_recovery.algebra import (
    NUM_SYNDROME_CLASSES,
    angle_table,
    build_recovery_model,
    verify_factorization,
)
from bb64_syndrome_recovery.controller import evaluate_policy, monte_carlo_policy


class RecoveryAlgebraTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = build_recovery_model()

    def test_complete_syndrome_enumeration(self) -> None:
        self.assertEqual(self.model.syndrome_coordinates.shape, (NUM_SYNDROME_CLASSES, 16))
        self.assertEqual(len(np.unique(self.model.base4_branch_id)), NUM_SYNDROME_CLASSES)
        self.assertEqual(int(self.model.base4_branch_id.min()), 0)
        self.assertEqual(int(self.model.base4_branch_id.max()), NUM_SYNDROME_CLASSES - 1)
        self.assertTrue(
            np.array_equal(
                self.model.displayed_syndromes[:, list(self.model.independent_syndrome_rows)],
                self.model.syndrome_coordinates,
            )
        )

    def test_kernel_is_exact_logical_triple_span(self) -> None:
        self.assertFalse(np.any((self.model.syndrome_matrix @ self.model.triple_generators) % 2))
        self.assertTrue(
            np.array_equal(
                (self.model.triple_generators.T @ self.model.piece_supports) % 2,
                self.model.code.logical_z,
            )
        )

    def test_every_canonical_correction_has_the_reported_syndrome(self) -> None:
        actual = (self.model.correction_physical_supports @ self.model.code.matrix_x.T) % 2
        self.assertTrue(np.array_equal(actual, self.model.displayed_syndromes))
        local_weights = np.count_nonzero(
            self.model.canonical_piece_masks.reshape(-1, 8, 3), axis=2
        )
        self.assertLessEqual(int(local_weights.max()), 1)
        reconstructed = self.model.initial_piece_masks ^ (
            self.model.logical_z_frame @ self.model.triple_generators.T
        ).astype(np.uint8)
        self.assertTrue(np.array_equal(reconstructed, self.model.canonical_piece_masks))

    def test_local_branch_angles(self) -> None:
        theta = math.pi / 32
        table = angle_table(theta)
        self.assertAlmostEqual(float(table["target"]["logical_angle"]), theta, places=14)
        self.assertAlmostEqual(
            float(table["alternative"]["logical_angle"]),
            float(table["physical_angle"]),
            places=14,
        )
        self.assertAlmostEqual(
            float(table["probability_target"]) + float(table["probability_any_alternative"]),
            1.0,
            places=14,
        )

    def test_factorization_certificate(self) -> None:
        certificate = verify_factorization(self.model, math.pi / 32, exhaustive=False)
        self.assertEqual(certificate["logical_support_error_count"], 0)
        self.assertEqual(certificate["logical_pairing_error_count"], 0)
        self.assertTrue(certificate["corrected_branches_are_product_rotations"])


class AdaptiveControllerTest(unittest.TestCase):
    def test_pi32_policy_benchmarks(self) -> None:
        expected = {
            0: (0.0497061496318901, 20.1182350152993),
            1: (0.230750772618601, 5.11826941935667),
            2: (0.519246374126126, 2.83014056086077),
            3: (0.781943131947136, 2.21529790536271),
            8: (1.0, 1.95029385036811),
        }
        for threshold, (keep, stages) in expected.items():
            result = evaluate_policy(math.pi / 32, threshold)
            self.assertAlmostEqual(result.keep_probability_per_attempt, keep, places=13)
            self.assertAlmostEqual(result.expected_total_tmr_stages, stages, places=13)
            self.assertLess(result.final_angle_error, 1e-14)

    def test_monte_carlo_matches_analytic(self) -> None:
        analytic = evaluate_policy(math.pi / 32, 3)
        sampled = monte_carlo_policy(math.pi / 32, 3, shots=200_000, seed=7)
        difference = abs(float(sampled["keep_probability"]) - analytic.keep_probability_per_attempt)
        self.assertLess(difference, 5 * float(sampled["keep_probability_se"]))


if __name__ == "__main__":
    unittest.main()
