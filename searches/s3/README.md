# Nonabelian S3 searches, folds, pairing, and resize studies

Run these tools as modules from the repository root. Shared algorithms live in
`gala_search/`; the existing default output locations under `results/` are retained.

## Reports

- [S3_FOLD_ANALYSIS.md](S3_FOLD_ANALYSIS.md)
- [S3_L6_NATIVE_RESULTS.md](S3_L6_NATIVE_RESULTS.md)
- [S3_LINEAR_FAST_FALSIFICATION.md](S3_LINEAR_FAST_FALSIFICATION.md)
- [S3_MANY_COPY_RESULTS.md](S3_MANY_COPY_RESULTS.md)
- [S3_RESULTS.md](S3_RESULTS.md)
- [S3_REVISIT_RESULTS.md](S3_REVISIT_RESULTS.md)
- [S3_SINGLE_GRID_RESULTS.md](S3_SINGLE_GRID_RESULTS.md)

## Commands

- `analyze_s3_candidate_folds.py`: `.venv/bin/python -m searches.s3.analyze_s3_candidate_folds`
- `analyze_s3_single_grid_candidate.py`: `.venv/bin/python -m searches.s3.analyze_s3_single_grid_candidate --help`
- `certify_s3_many_copy.py`: `.venv/bin/python -m searches.s3.certify_s3_many_copy --help`
- `combine_s3_many_copy_fold_grids.py`: `.venv/bin/python -m searches.s3.combine_s3_many_copy_fold_grids --help`
- `enumerate_s3_many_copy_fold_grids.py`: `.venv/bin/python -m searches.s3.enumerate_s3_many_copy_fold_grids --help`
- `revisit_s3_joint_pairing.py`: `.venv/bin/python -m searches.s3.revisit_s3_joint_pairing --help`
- `revisit_s3_saved_batch.py`: `.venv/bin/python -m searches.s3.revisit_s3_saved_batch --help`
- `run_s3_ising_search.py`: `.venv/bin/python -m searches.s3.run_s3_ising_search --help`
- `run_s3_linear_l6_local.py`: `.venv/bin/python -m searches.s3.run_s3_linear_l6_local --help`
- `run_s3_linear_l6_native.py`: `.venv/bin/python -m searches.s3.run_s3_linear_l6_native --help`
- `run_s3_linear_relift.py`: `.venv/bin/python -m searches.s3.run_s3_linear_relift --help`
- `run_s3_linear_resize_pilots.py`: `.venv/bin/python -m searches.s3.run_s3_linear_resize_pilots --help`
- `run_s3_many_copy_fold_search.py`: `.venv/bin/python -m searches.s3.run_s3_many_copy_fold_search --help`
- `run_s3_many_copy_search.py`: `.venv/bin/python -m searches.s3.run_s3_many_copy_search --help`
- `run_s3_single_grid_search.py`: `.venv/bin/python -m searches.s3.run_s3_single_grid_search --help`
- `screen_s3_linear_distance.py`: `.venv/bin/python -m searches.s3.screen_s3_linear_distance --help`
- `screen_s3_linear_grid_sets.py`: `.venv/bin/python -m searches.s3.screen_s3_linear_grid_sets --help`
- `screen_s3_linear_resize_distance.py`: `.venv/bin/python -m searches.s3.screen_s3_linear_resize_distance --help`
- `screen_s3_linear_resize_grid_sets.py`: `.venv/bin/python -m searches.s3.screen_s3_linear_resize_grid_sets --help`
- `screen_s3_many_copy_fold.py`: `.venv/bin/python -m searches.s3.screen_s3_many_copy_fold --help`
- `search_s3_many_copy_fold_grids.py`: `.venv/bin/python -m searches.s3.search_s3_many_copy_fold_grids --help`
