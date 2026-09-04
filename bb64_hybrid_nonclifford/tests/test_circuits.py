"""Structural and noiseless tests for hybrid repair circuits."""

from __future__ import annotations

import math
import unittest

import numpy as np

from bb64_hybrid_nonclifford.actions import repair_action, verify_action_angles
from bb64_hybrid_nonclifford.circuit import (
    build_ideal_branch_repair_circuit,
    build_synthetic_repair_circuit,
    class_id_for_labels,
)
from bb64_hybrid_nonclifford.decoder import NoiselessTableDecoder
from bb64_hybrid_nonclifford.model import load_hybrid_model
from bb64_hybrid_nonclifford.raw_branch_calibration import (
    build_raw_branch_calibration_circuit,
)
from bb64_hybrid_nonclifford.repair_calibration import (
    build_repair_calibration_circuit,
)
from bb64_hybrid_nonclifford.syndrome_history import (
    build_syndrome_history_circuit,
    extract_syndrome_histories,
)
from bb64_hybrid_nonclifford.toy_tmr import build_toy_branch_circuit
from bb64_tmr_postselection.circuit import NoiseModel


class HybridCircuitTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = load_hybrid_model()

    def test_branch_label_lookup(self) -> None:
        for labels in ((0,) * 8, (1, 0, 0, 0, 0, 0, 0, 0), (1, 2, 3, 0, 0, 0, 0, 0)):
            class_id = class_id_for_labels(self.model, labels)
            self.assertEqual(tuple(self.model.recovery.local_branch_labels[class_id]), labels)

    def test_ideal_branch_bundle(self) -> None:
        class_id = class_id_for_labels(self.model, (1, 0, 0, 0, 0, 0, 0, 0))
        bundle = build_ideal_branch_repair_circuit(
            self.model, theta=math.pi / 32, syndrome_class_id=class_id
        )
        self.assertEqual(bundle.initial_detector_count, 64)
        self.assertEqual(bundle.final_detector_count, 64)
        self.assertEqual(sum(bundle.postselection_mask), 64)
        self.assertEqual(bundle.observable_count, 8)
        self.assertEqual(bundle.operation_counts["repair_rotations"], 1)

    def test_synthetic_bundle_has_one_repair_per_alternative(self) -> None:
        bundle = build_synthetic_repair_circuit(
            self.model,
            theta=math.pi / 32,
            alternative_mask=(1, 0, 1, 0, 0, 0, 0, 0),
        )
        self.assertEqual(bundle.operation_counts["repair_rotations"], 2)
        self.assertEqual(bundle.observable_count, 8)

    def test_repair_calibration_parallel_layer_accounting(self) -> None:
        one = build_repair_calibration_circuit(
            self.model,
            theta=math.pi / 32,
            alternative_mask=(1, 0, 0, 0, 0, 0, 0, 0),
            noise=NoiseModel(),
        )
        same_batch = build_repair_calibration_circuit(
            self.model,
            theta=math.pi / 32,
            alternative_mask=(1, 0, 0, 1, 0, 0, 0, 0),
            noise=NoiseModel(),
        )
        split_batches = build_repair_calibration_circuit(
            self.model,
            theta=math.pi / 32,
            alternative_mask=(1, 1, 0, 0, 0, 0, 0, 0),
            noise=NoiseModel(),
        )
        self.assertEqual(one.operation_counts["cnot_layers"], 22)
        self.assertEqual(same_batch.operation_counts["cnot_layers"], 22)
        self.assertEqual(split_batches.operation_counts["cnot_layers"], 36)
        self.assertEqual(one.operation_counts["physical_cnots"], 526)
        self.assertEqual(same_batch.operation_counts["physical_cnots"], 540)

    def test_raw_branch_expected_detector_shape(self) -> None:
        class_id = class_id_for_labels(self.model, (1, 0, 0, 0, 0, 0, 0, 0))
        bundle = build_raw_branch_calibration_circuit(
            self.model,
            theta=math.pi / 32,
            syndrome_class_id=class_id,
            noise=NoiseModel(),
        )
        self.assertEqual(len(bundle.postselection_mask), len(bundle.expected_detectors))
        self.assertEqual(sum(bundle.postselection_mask), 128)
        self.assertEqual(bundle.operation_counts["repaired_logicals"], 1)

    def test_frame_aware_repair_action(self) -> None:
        class_id = class_id_for_labels(self.model, (1, 0, 2, 0, 0, 0, 0, 0))
        decoded = NoiselessTableDecoder(self.model).decode(self.model.displayed_syndromes[class_id])
        action = repair_action(
            decoded,
            theta=math.pi / 32,
            threshold=2,
            logical_x_frame=(1, 0, 0, 0, 0, 0, 0, 0),
        )
        self.assertFalse(action.reset)
        self.assertLess(action.signed_residual_angles[0], 0)
        self.assertGreater(action.signed_residual_angles[2], 0)
        self.assertLess(verify_action_angles(decoded, action, theta=math.pi / 32), 1e-14)
        rejected = repair_action(decoded, theta=math.pi / 32, threshold=1)
        self.assertTrue(rejected.reset)
        self.assertEqual(rejected.reset_reason, "repair_threshold")

    def test_toy_branch_probabilities_sum_to_one(self) -> None:
        bundles = [build_toy_branch_circuit(theta=math.pi / 32, branch_label=i) for i in range(4)]
        self.assertAlmostEqual(sum(bundle.expected_probability for bundle in bundles), 1.0, places=14)

    def test_retained_syndrome_history_groups(self) -> None:
        bundle = build_syndrome_history_circuit(
            self.model,
            theta=math.pi / 32,
            noise=NoiseModel(),
            syndrome_rounds=3,
        )
        self.assertEqual(tuple(map(len, bundle.x_detector_groups)), (32, 32, 32))
        self.assertEqual(tuple(map(len, bundle.z_detector_groups)), (32, 32, 32))
        self.assertEqual(bundle.operation_counts["cnot_layers"], 40)
        detector_width = 1 + max(
            index
            for group in (*bundle.x_detector_groups, *bundle.z_detector_groups)
            for index in group
        )
        detectors = np.zeros((2, detector_width), dtype=np.uint8)
        detectors[1, bundle.x_detector_groups[1][7]] = 1
        x_history, z_history = extract_syndrome_histories(detectors, bundle)
        self.assertEqual(x_history.shape, (2, 3, 32))
        self.assertEqual(z_history.shape, (2, 3, 32))
        self.assertEqual(x_history[1, 1, 7], 1)


if __name__ == "__main__":
    unittest.main()
