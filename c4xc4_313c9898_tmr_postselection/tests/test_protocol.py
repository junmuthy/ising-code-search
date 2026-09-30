import math
from pathlib import Path
from tempfile import TemporaryDirectory
import time
import unittest

import clifft
import numpy as np
import stim

from bicycle_chain_l2_paper.tmr_postselection import circuit as engine
from bicycle_chain_l2_paper.tmr_postselection.model import initialization_correction_map, row_relations
from c4xc4_313c9898_tmr_postselection.model import load_code_artifact, validate_schedule
from c4xc4_313c9898_tmr_postselection.protocol import (
    acceptance_masks, build_circuit, exact_output_audit, partition_certificate,
    physical_tmr_angle, selected_logicals, verification_tail,
)
from c4xc4_313c9898_tmr_postselection.run import run_point, wilson


class ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.codes = {p: load_code_artifact(p) for p in ("preferred", "original")}

    def test_frozen_artifacts_all_logicals_retained(self):
        for p, code in self.codes.items():
            self.assertEqual(code.num_logicals, 16)
            self.assertEqual(code.num_checks, 24 if p == "preferred" else 32)
            self.assertEqual(code.batches, ((0, 2, 4, 6),))
            self.assertEqual(len(row_relations(code.matrix_z)), 0 if p == "preferred" else 8)

    def test_changed_hash_rejected(self):
        with TemporaryDirectory() as temp:
            target = Path(temp)/"preferred"
            target.mkdir()
            (target/"code.json").write_text("{}")
            (target/"schedule.json").write_text("{}")
            with self.assertRaises(ValueError):
                load_code_artifact(root=Path(temp))

    def test_scope_restriction(self):
        for n in (0, 2, 3, 8, 16):
            with self.assertRaises(ValueError):
                selected_logicals(n)

    def test_schedule_missing_edge_rejected(self):
        code = self.codes["preferred"]
        bad = (code.schedule[0][1:],) + code.schedule[1:]
        with self.assertRaises(ValueError):
            validate_schedule(code.matrix_x, code.matrix_z, bad)

    def test_certificates(self):
        for code in self.codes.values():
            for n in (1, 4):
                cert = partition_certificate(code, n)
                self.assertEqual(cert["partial_syndrome_rank"], 2*n)
                self.assertEqual(cert["kernel_dimension"], n)

    def test_exact_output_sign_and_inactive_logicals(self):
        for code in self.codes.values():
            for n in (1, 4):
                for theta in (0, .001, math.pi/128, math.pi/32, -math.pi/32):
                    audit = exact_output_audit(code, n, theta)
                    self.assertLess(audit["max_output_amplitude_error"], 1e-12)
                    self.assertTrue(audit["inactive_logicals_unchanged"])
        self.assertLess(physical_tmr_angle(math.pi/32, 3), 0)

    def test_initialization_feedback(self):
        for code in self.codes.values():
            rows, qubits, right = initialization_correction_map(code.matrix_z)
            self.assertEqual(len(rows), 24)
            self.assertEqual(len(qubits), 24)
            np.testing.assert_array_equal(code.matrix_z[list(rows)] @ right % 2, np.eye(24))

    def test_encoder(self):
        code = self.codes["preferred"]
        simulator = stim.TableauSimulator()
        simulator.do_circuit(engine.encoded_plus_circuit(code))
        for axis, supports in (("X", code.checks_x), ("Z", code.checks_z), ("X", code.logicals_x)):
            for support in supports:
                pauli = stim.PauliString(64)
                for q in support:
                    pauli[q] = axis
                self.assertEqual(simulator.peek_observable_expectation(pauli), 1)

    def test_operation_counts_and_masks(self):
        for p, code in self.codes.items():
            for n in (1, 4):
                bundle = build_circuit(code, n, probability=.001)
                self.assertEqual(bundle.operation_counts["physical_cnots"], 36*code.num_checks+6*n)
                self.assertEqual(bundle.operation_counts["cnot_layers"], 26)
                self.assertEqual(bundle.operation_counts["physical_rotations"], 3*n)
                self.assertEqual(bundle.operation_counts["rotation_layers"], 1)
                self.assertEqual(sum(bundle.postselection_mask), 2*code.num_checks)
                relations = bundle.detector_groups.get("initialization_relations", ())
                self.assertTrue(all(bundle.postselection_mask[i] == 0 for i in relations))
                self.assertEqual(bundle.operation_counts["idle_locations"], 24*64-36*code.num_checks+128-12*n)

    def test_paired_policy_logic(self):
        # Initial relation failures are diagnostic, not rejection, for this baseline.
        d = np.array([[1,0,0], [0,1,0], [0,0,1], [0,1,1]], dtype=bool)
        masks = acceptance_masks(d, {"initialization_relations": (0,), "post_X": (1,), "post_Z": (2,)})
        self.assertEqual(masks["tmr-x"].tolist(), [True,False,True,False])
        self.assertEqual(masks["strict-xz"].tolist(), [True,False,False,False])

    def test_zero_angle_stim_and_clifft(self):
        for code in self.codes.values():
            bundle = build_circuit(code, 4, theta=0.)
            # Independently test the Clifford zero-angle circuit using Stim.
            text = "\n".join(line for line in bundle.text.splitlines() if not line.startswith("R_Z("))
            d = stim.Circuit(text).compile_detector_sampler(seed=19).sample(512)
            self.assertFalse(np.any(d))
            sample = clifft.sample(clifft.compile(bundle.text), shots=512, seed=19)
            self.assertTrue(np.all(acceptance_masks(sample.detectors, bundle.detector_groups)["strict-xz"]))

    def test_full_nonzero_output_and_negative_control(self):
        for code in self.codes.values():
            bundle = build_circuit(code, 4, theta=math.pi/8)
            for sign in (-1, 1):
                text, probes = verification_tail(bundle, code, sign)
                sample = clifft.sample(clifft.compile(text), shots=4000, seed=140+sign, threads=4)
                accepted = acceptance_masks(sample.detectors, bundle.detector_groups)["strict-xz"]
                failures = np.any(sample.detectors[accepted][:, list(probes)], axis=1)
                self.assertGreater(int(np.sum(accepted)), 50)
                if sign == -1:
                    self.assertFalse(np.any(failures))
                else:
                    self.assertGreater(int(np.sum(failures)), 10)

    def test_noisy_zero_angle_independent_stim_crosscheck(self):
        shots = 100_000
        for code in self.codes.values():
            for n in (1, 4):
                bundle = build_circuit(code, n, theta=0., probability=.001)
                # Remove only the zero-angle identity. Keep its rotation noise,
                # all ladder CNOTs, feedback, and entangling-layer idle noise.
                text = "\n".join(line for line in bundle.text.splitlines() if not line.startswith("R_Z("))
                stim_d = stim.Circuit(text).compile_detector_sampler(seed=445+n).sample(shots)
                clifft_d = clifft.sample(clifft.compile(bundle.text), shots=shots, seed=774+n, threads=4).detectors
                left = acceptance_masks(stim_d, bundle.detector_groups)
                right = acceptance_masks(clifft_d, bundle.detector_groups)
                for policy in ("strict-xz", "tmr-x"):
                    p, q = float(np.mean(left[policy])), float(np.mean(right[policy]))
                    se = math.sqrt((p*(1-p)+q*(1-q))/shots)
                    self.assertLess(abs(p-q), 6*se)

    def test_noisy_fidelity_tail_rejected(self):
        code = self.codes["preferred"]
        with self.assertRaises(ValueError):
            verification_tail(build_circuit(code, 1, probability=.001), code)

    def test_m1_control(self):
        code = self.codes["preferred"]
        bundle = build_circuit(code, 4, partition_count=1)
        self.assertAlmostEqual(bundle.ideal_acceptance, 1)
        self.assertEqual(bundle.operation_counts["physical_cnots"], 864+40)
        text, probes = verification_tail(bundle, code)
        sample = clifft.sample(clifft.compile(text), shots=200, seed=991)
        self.assertFalse(np.any(sample.detectors))

    def test_resume_and_configuration_guard(self):
        code = self.codes["preferred"]
        config = {"logical_count": 1, "theta": 0., "probability": 0., "mode": "full", "partition_count": 3}
        with TemporaryDirectory() as temp:
            path = Path(temp)/"run.json"
            options = dict(shots=100, checkpoint_shots=50, threads=1, seed=777, hashes={"test": "fixed"})
            interrupted = run_point(path, code, config, deadline=time.monotonic()-1, **options)
            self.assertFalse(interrupted["completed"])
            result = run_point(path, code, config, deadline=time.monotonic()+60, resume=True, **options)
            self.assertTrue(result["completed"])
            self.assertEqual(result["counts"]["attempted"], 100)
            self.assertEqual(result["counts"]["accepted"]["strict-xz"], 100)
            with self.assertRaises(ValueError):
                run_point(path, code, {**config, "theta": .001}, deadline=time.monotonic()+60, resume=True, **options)
            with self.assertRaises(ValueError):
                run_point(path, code, config, deadline=time.monotonic()+60, **options)

    def test_wilson(self):
        lo, hi = wilson(500, 1000)
        self.assertLess(lo, .5)
        self.assertGreater(hi, .5)


if __name__ == "__main__":
    unittest.main()
