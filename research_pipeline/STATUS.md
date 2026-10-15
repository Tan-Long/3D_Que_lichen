# Project status

Last verified: 2026-10-12 on macOS.

## Current state

- Production is frozen at `baseline-v1-20261012`, using only
  `tensor_v5mm_step3_conf1_adaptive_r35`. Experimental observed-boost and
  frame-consistency outputs are excluded. The authoritative downstream input is
  `manifests/production_baseline_v1_approved.csv`: 137 approved geometry samples.
  Surface-area measurement is ready for 135; `cay_0016_2305_TL6` and
  `TV6__video_20260630_135554` remain measurement-deferred because terrain
  direction is undefined, and the pipeline correctly refuses to invent it.

- Raw dataset: 306 scans; all manifest paths resolve on this machine.
- Geometry-usable canonical set: 299 scans.
- Full r35 reconstruction: complete for all 299 canonical geometry scans: 293 `ok`, 4 mesh-complete/measurement-deferred, and 2 raw-insufficient.
- Observed-point diagnostic: complete for all 299 canonical scans: 296 built, 1 pre-existing, and 2 skipped because their raw baseline is insufficient.
- Observed-support TSDF boost pilot: closed and packaged under `artifacts/reconstruction_pipeline_pilot/`. It does not replace the production baseline; no pilot mesh has been promoted.
- Selected observed-boost batch: complete on 145 scans. Results: 142 newly built, 1 existing pilot, 2 raw-insufficient, and no build failures. QA pass changed from 84 baseline to 82 boost, so the experiment was not adopted globally.
- Vision QA deep dive: all 140 manually approved baseline previews and all 145 available baseline/boost render pairs were inspected. None of the four metric-only deferred-to-pass candidates survived visual promotion; 12 approved samples need clear re-review and 4 more are on a double-shell/pose-drift watchlist. Details are in `artifacts/vision_qa/VISION_QA_DEEP_DIVE.md`.
- Non-approved vision audit: all 159 unique `needs_review`/`rejected` samples were re-checked (157 rendered previews plus 2 raw-insufficient cases). No rejected sample should be reopened; 47 current `needs_review` samples are recommended for rejection and 13 visually near-pass samples should remain under review. No user decision was changed. Details are in `artifacts/vision_qa/NONAPPROVED_VISION_AUDIT.md`.
- Double-shell consistency pilot: an opt-in measured-radius frame filter was implemented and tested on `26Q90`, `26Q20`, and `cay_0029_1805`. It partly improved `26Q90` but regressed coverage or residuals on the other cases, so no output was promoted and no batch run was started. See `artifacts/frame_consistency_pilot/REPORT.md`.
- Semantic work: three-scan HSV baseline only; rejected as a biological measurement pending reference labels.

The authoritative full-run state is `artifacts/meshes_all/tensor_v5mm_step3_conf1_adaptive_r35_all_run_log.json`.
The observed-point run state is `artifacts/observed_points_all/v5mm_step12_pixel2_conf1_r22_run_log.json`.
The current macOS console log is `artifacts/observed_points_all/observed_points_resume_macos.log`.
The selected boost run state is `artifacts/reconstruction_pipeline_pilot/tensor_v5mm_step3_conf1_adaptive_r35_observed_boost_v1_selected_run_log.json`.
Its console log is `artifacts/reconstruction_pipeline_pilot/observed_boost_selected_resume_macos.log`.

## Check progress

```bash
jq 'group_by(.status) | map({status: .[0].status, count: length})' \
  artifacts/observed_points_all/v5mm_step12_pixel2_conf1_r22_run_log.json
```

## Resume safely

```bash
.venv/bin/python tools/run_observed_points_all.py \
  --workers 2 --frame-step 12 --pixel-step 2
```

Resume the selected boost batch without changing production QA manifests:

```bash
.venv/bin/python tools/run_tsdf_recovery.py --workers 2 --frame-step 3 \
  --roi-search-radius 0.35 --output-root artifacts/reconstruction_pipeline_pilot \
  --observed-support-root artifacts/observed_points_all \
  --include-low-support-below 0.5 --skip-classify
```

The Python environment is stored in `.venv` on the NTFS volume. Two workers are the tested limit for this 16 GB Mac. Keep the volume mounted read-write with Paragon, prevent Mac sleep, and do not disconnect it while a job is running.
