"""Reuse the audited physical replay and exact signature engines unchanged."""
from __future__ import annotations

import argparse
import importlib.util

from model import SHARED

spec = importlib.util.spec_from_file_location('d6_shared_fault_tools', SHARED / 'faults.py')
shared = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shared)
for name in ('replay', 'explain_masks', 'bounded_heuristic', 'heuristic_worker',
             'certify_through_three', 'certify_through_four', 'estimate_four_fault_table',
             'extract_fault_effects', 'detector_error_model', 'inventory'):
    globals()[name] = getattr(shared, name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--heuristic', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    heuristic_worker(args.heuristic, args.output)
