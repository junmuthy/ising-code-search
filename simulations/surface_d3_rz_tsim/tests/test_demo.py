from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from circuit import (  # noqa: E402
    CNOT_LAYERS,
    build_projection_circuit,
    build_scheduled_circuit,
    ideal_tmr_success,
    physical_tmr_angle,
)
from surface_code import (  # noqa: E402
    X_ANCILLAS,
    X_CHECKS,
    Z_ANCILLAS,
    Z_CHECKS,
    validate_code,
)


class SurfaceCodeTest(unittest.TestCase):
    def test_exact_code_and_encoder_validation(self) -> None:
        report = validate_code()
        self.assertEqual((report["n"], report["k"], report["distance"]), (9, 1, 3))
        self.assertTrue(report["checks_commute"])
        self.assertTrue(report["encoder_valid"])

    def test_m3_angle_includes_required_sign(self) -> None:
        for theta in (-0.4, -0.01, 0.0, 0.01, 0.4):
            theta_star = physical_tmr_angle(theta, 3)
            logical_tangent = -math.tan(theta_star / 2) ** 3
            self.assertAlmostEqual(logical_tangent, math.tan(theta / 2), places=13)

    def test_success_probability_forms_agree(self) -> None:
        for theta in (-0.4, 0.0, 0.01, 0.4):
            theta_star = physical_tmr_angle(theta, 3)
            direct = math.cos(theta_star / 2) ** 6 + math.sin(theta_star / 2) ** 6
            self.assertAlmostEqual(ideal_tmr_success(theta, 3), direct, places=14)

    def test_circuit_metadata(self) -> None:
        projection = build_projection_circuit(0.01, "Y")
        scheduled = build_scheduled_circuit(0.01, "Z")
        self.assertEqual(projection.circuit.num_detectors, 8)
        self.assertEqual(scheduled.circuit.num_detectors, 32)
        self.assertEqual(len(scheduled.detector_groups["initialization"]), 16)
        self.assertEqual(len(scheduled.detector_groups["post_tmr"]), 16)
        self.assertEqual(projection.circuit.num_observables, 1)
        self.assertEqual(scheduled.circuit.num_observables, 1)

    def test_cnot_schedule_measures_declared_checks(self) -> None:
        measured_x = {
            ancilla: set() for ancilla in X_ANCILLAS
        }
        measured_z = {
            ancilla: set() for ancilla in Z_ANCILLAS
        }
        for layer in CNOT_LAYERS:
            touched: set[int] = set()
            for control, target in layer:
                self.assertNotIn(control, touched)
                self.assertNotIn(target, touched)
                touched.update((control, target))
                if control in measured_x:
                    measured_x[control].add(target)
                if target in measured_z:
                    measured_z[target].add(control)
        self.assertEqual(
            {frozenset(support) for support in measured_x.values()},
            {frozenset(support) for support in X_CHECKS},
        )
        self.assertEqual(
            {frozenset(support) for support in measured_z.values()},
            {frozenset(support) for support in Z_CHECKS},
        )

    def test_identity_projection_is_deterministic(self) -> None:
        bundle = build_projection_circuit(0.0, "X")
        detectors, observables = bundle.circuit.compile_detector_sampler(seed=9).sample(
            64,
            batch_size=64,
            separate_observables=True,
        )
        self.assertFalse(np.asarray(detectors).any())
        self.assertFalse(np.asarray(observables).any())


if __name__ == "__main__":
    unittest.main()
