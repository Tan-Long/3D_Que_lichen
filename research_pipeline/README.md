# Current research pipeline

This directory contains the code and lightweight documentation used for the current cinnamon-trunk reconstruction and lichen-analysis experiments. Datasets, reconstructed meshes, model weights, generated manifests, and QA artifacts are deliberately not included.

## Pipeline

1. Audit Stray Scanner captures and build a local manifest.
2. Reconstruct observed trunk geometry with Open3D Tensor `VoxelBlockGrid` TSDF.
3. Run geometry QA and visually review one complete mesh per tree.
4. Extract DINOv3 ViT-L features, cluster surface appearance, and assign human material labels.
5. Probe the resulting labels against reviewed trees before reporting surface coverage.

Clusters are visual groups, not biological classes. The current material probe is experimental and its predictions must be reviewed before use in ecological analysis.

## Expected local data

Reconstruction tools expect each scan to provide RGB video, camera intrinsics and odometry, depth frames, and confidence frames. Exact filenames and manifest fields are documented in [`manifests/README.md`](manifests/README.md).

Generated manifests can contain absolute paths and precise capture locations. Keep them local unless they have been sanitized.

## Environment

The scripts target Python 3.11 and use packages including NumPy, SciPy, OpenCV, Pillow, Open3D, and PyTorch. DINOv3 source code and licensed weights must be obtained separately from the official project; they are not vendored here.

Start with [`STATUS.md`](STATUS.md) for the current baseline and [`RECONSTRUCTION_RESEARCH_PLAN.md`](RECONSTRUCTION_RESEARCH_PLAN.md) for the alternatives already evaluated.
