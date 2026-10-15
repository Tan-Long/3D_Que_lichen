# Que lichen

Research tooling and decision tracking for reconstructing cinnamon-tree trunk surfaces from Stray Scanner RGB-D captures, segmenting epiphytic organisms, and measuring observed surface coverage.

## Current geometry pipeline

- Open3D Tensor `VoxelBlockGrid` TSDF.
- Per-frame camera intrinsics and poses.
- Confidence and terrain-aware trunk ROI filtering.
- Held-out depth QA and manual mesh review.
- Surface measurements over observed geometry only; holes are not silently filled.

See the [research pipeline guide](research_pipeline/README.md), [project status](research_pipeline/STATUS.md), and [reconstruction research plan](research_pipeline/RECONSTRUCTION_RESEARCH_PLAN.md).

## Repository contents

- `research_pipeline/tools/`: current dataset audit, TSDF reconstruction, QA, diagnostics, DINOv3 clustering, measurement, and review utilities.
- `research_pipeline/prototypes/upslope_downslope/`: terrain-direction prototype.
- `research_pipeline/manifests/README.md`: manifest schema and regeneration notes.
- Repository-root scripts: earlier pipeline retained for reproducibility and comparison.

## Data policy

Raw captures, meshes, browser profiles, generated manifests, and QA artifacts are intentionally excluded. Generated manifests contain local paths and precise capture locations and must be reviewed or sanitized before publication.

## Status

The repository is research-stage. Geometry reconstruction and QA are active; the HSV semantic baseline is retained only for regression and is not a biological measurement.
