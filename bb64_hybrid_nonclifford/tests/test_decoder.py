"""Regression tests for the branch-aware decoder."""

from __future__ import annotations

import math
import unittest

import numpy as np

from bb64_hybrid_nonclifford.decoder import (
    MeasurementMapDecoder,
    NoiselessTableDecoder,
    sample_measurement_histories,
)
from bb64_hybrid_nonclifford.model import load_hybrid_model


class HybridModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = load_hybrid_model()

    def test_quotient_coordinates_are_linear_and_detect_alternatives(self) -> None:
        predicted = (
            self.model.recovery.syndrome_coordinates @ self.model.syndrome_to_quotient.T
        ) % 2
        self.assertTrue(np.array_equal(predicted, self.model.quotient_coordinates.reshape(-1, 16)))
        alternatives = np.any(self.model.quotient_coordinates, axis=2)
        self.assertTrue(np.array_equal(alternatives, self.model.recovery.alternative_mask))

    def test_noiseless_decoder_rejects_out_of_image_syndrome(self) -> None:
        decoder = NoiselessTableDecoder(self.model)
        syndrome = self.model.displayed_syndromes[1234].copy()
        decoded = decoder.decode(syndrome)
        self.assertEqual(decoded.syndrome_class_id, 1234)
        syndrome[31] ^= 1
        with self.assertRaises(ValueError):
            decoder.decode(syndrome)

    def test_all_class_coordinates_round_trip(self) -> None:
        rows = list(self.model.recovery.independent_syndrome_rows)
        ids = self.model.class_ids_from_coordinates(self.model.displayed_syndromes[:, rows])
        self.assertTrue(np.array_equal(ids, np.arange(65536, dtype=np.uint32)))


class MeasurementMapDecoderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = load_hybrid_model()

    def test_clean_repeated_histories_decode_exactly(self) -> None:
        ids = np.array([0, 1, 2, 3, 17, 101, 4095, 32768, 65535], dtype=np.uint32)
        histories = np.repeat(self.model.displayed_syndromes[ids, None, :], 3, axis=1)
        decoder = MeasurementMapDecoder(
            self.model,
            theta=math.pi / 32,
            measurement_probability=0.01,
        )
        result = decoder.decode_batch(histories, batch_size=3)
        self.assertTrue(np.array_equal(result.syndrome_class_id, ids))
        self.assertTrue(np.all(result.likelihood_gap > 0))

    def test_repetition_suppresses_readout_mistakes(self) -> None:
        rng = np.random.default_rng(5)
        ids = rng.integers(0, 65536, size=100, dtype=np.uint32)
        histories = sample_measurement_histories(
            self.model,
            ids,
            rounds=5,
            measurement_probability=0.01,
            seed=9,
        )
        decoder = MeasurementMapDecoder(
            self.model,
            theta=math.pi / 32,
            measurement_probability=0.01,
        )
        result = decoder.decode_batch(histories, batch_size=10)
        self.assertGreaterEqual(float(np.mean(result.syndrome_class_id == ids)), 0.99)


if __name__ == "__main__":
    unittest.main()
