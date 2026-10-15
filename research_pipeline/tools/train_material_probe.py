#!/usr/bin/env python3
"""Train a lightweight bark-to-lichen score from reviewed DINOv3-L K=3 clusters."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import lsqr


ROOT = Path(__file__).resolve().parents[1]
PILOT_ROOT = ROOT / "artifacts/surface_cluster_pilot"
REVIEWS_PATH = PILOT_ROOT / "cluster_reviews_dinov3_k3.json"
SOURCE_VARIANTS = ("dinov3_vitl16_k3", "dinov3_vitl16_k3_cached")
OUTPUT_VARIANT = "material_probe_pure_v2"
PURE_TARGETS = {
    "bark": 0.0,
    "lichen": 1.0,
}


@dataclass
class TreeData:
    uid: str
    source_dir: Path
    labels: np.ndarray
    observations: np.ndarray
    areas_m2: np.ndarray
    projected: np.ndarray
    pca_mean: np.ndarray
    pca_components: np.ndarray
    reviews: dict[int, str]

    @property
    def observed_indices(self) -> np.ndarray:
        return np.flatnonzero(self.labels >= 0)


def reconstruct_features(projected: np.ndarray, mean: np.ndarray,
                         components: np.ndarray) -> np.ndarray:
    features = projected @ components + mean
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return np.divide(features, norms, out=np.zeros_like(features), where=norms > 0)


def source_dir_for(uid: str, root: Path = PILOT_ROOT) -> Path:
    for variant in SOURCE_VARIANTS:
        source_dir = root / uid / variant
        if (source_dir / "surface_cluster_triangle_data.npz").is_file():
            return source_dir
    raise FileNotFoundError(f"{uid}: no DINOv3-L K=3 triangle data found")


def load_trees(reviews_path: Path = REVIEWS_PATH) -> list[TreeData]:
    review_data = json.loads(reviews_path.read_text(encoding="utf-8")).get("reviews", {})
    trees = []
    for uid, cluster_reviews in sorted(review_data.items()):
        source_dir = source_dir_for(uid)
        with np.load(source_dir / "surface_cluster_triangle_data.npz") as data:
            tree = TreeData(
                uid=uid,
                source_dir=source_dir,
                labels=np.asarray(data["labels"], dtype=np.int16),
                observations=np.asarray(data["observations"], dtype=np.uint16),
                areas_m2=np.asarray(data["areas_m2"], dtype=np.float64),
                projected=np.asarray(data["projected_features"], dtype=np.float32),
                pca_mean=np.asarray(data["pca_mean"], dtype=np.float32),
                pca_components=np.asarray(data["pca_components"], dtype=np.float32),
                reviews={int(cluster): review["decision"]
                         for cluster, review in cluster_reviews.items()},
            )
        if len(tree.observed_indices) != len(tree.projected):
            raise ValueError(f"{uid}: projected features do not match observed triangles")
        if set(tree.reviews) != set(range(3)):
            raise ValueError(f"{uid}: all three K=3 clusters must be reviewed")
        trees.append(tree)
    if len(trees) < 2:
        raise ValueError("at least two fully reviewed trees are required")
    return trees


def training_rows(trees: list[TreeData], sample_per_cluster: int, seed: int,
                  excluded_uid: str | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    feature_rows, targets, weights = [], [], []
    for tree in trees:
        if tree.uid == excluded_uid:
            continue
        eligible = [(cluster, PURE_TARGETS[decision])
                    for cluster, decision in tree.reviews.items()
                    if decision in PURE_TARGETS]
        observed_labels = tree.labels[tree.observed_indices]
        for cluster, target in eligible:
            indices = np.flatnonzero(observed_labels == cluster)
            if len(indices) > sample_per_cluster:
                indices = rng.choice(indices, sample_per_cluster, replace=False)
            feature_rows.append(reconstruct_features(
                tree.projected[indices], tree.pca_mean, tree.pca_components))
            targets.append(np.full(len(indices), target, dtype=np.float32))
            weights.append(np.full(len(indices), 1 / (len(eligible) * len(indices)),
                                   dtype=np.float32))
    if not feature_rows:
        raise ValueError("no bark/lichen supervision found")
    x = np.vstack(feature_rows)
    y = np.concatenate(targets)
    sample_weights = np.concatenate(weights)
    sample_weights *= len(sample_weights) / sample_weights.sum()
    if np.ptp(y) == 0:
        raise ValueError("training split needs both bark-leaning and lichen-leaning labels")
    return x, y, sample_weights


def fit_probe(features: np.ndarray, targets: np.ndarray, sample_weights: np.ndarray,
              ridge: float) -> np.ndarray:
    design = np.column_stack((features, np.ones(len(features), dtype=np.float32)))
    root_weights = np.sqrt(sample_weights)[:, None]
    result = lsqr(design * root_weights, targets * root_weights[:, 0],
                  damp=np.sqrt(ridge), atol=1e-5, btol=1e-5, iter_lim=200)
    return np.asarray(result[0], dtype=np.float32)


def predict_tree(tree: TreeData, coefficients: np.ndarray,
                 chunk_size: int = 8192) -> np.ndarray:
    scores = np.full(len(tree.labels), np.nan, dtype=np.float32)
    observed = tree.observed_indices
    for start in range(0, len(observed), chunk_size):
        stop = min(start + chunk_size, len(observed))
        features = reconstruct_features(
            tree.projected[start:stop], tree.pca_mean, tree.pca_components)
        scores[observed[start:stop]] = np.clip(
            features @ coefficients[:-1] + coefficients[-1], 0, 1)
    for cluster, decision in tree.reviews.items():
        if decision == "vegetation":
            scores[tree.labels == cluster] = np.nan
    return scores


def cluster_evaluation(tree: TreeData, scores: np.ndarray) -> list[dict]:
    rows = []
    for cluster, decision in sorted(tree.reviews.items()):
        if decision not in PURE_TARGETS:
            continue
        mask = (tree.labels == cluster) & np.isfinite(scores)
        predicted = float(np.average(scores[mask], weights=tree.areas_m2[mask]))
        target = PURE_TARGETS[decision]
        rows.append({"scan_uid": tree.uid, "cluster": cluster, "decision": decision,
                     "target": target, "predicted": predicted,
                     "absolute_error": abs(predicted - target)})
    return rows


def probability_colors(scores: np.ndarray, vegetation: np.ndarray) -> np.ndarray:
    bark = np.asarray([0.34, 0.16, 0.06])
    middle = np.asarray([0.55, 0.55, 0.50])
    lichen = np.asarray([0.95, 0.84, 0.22])
    colors = np.full((len(scores), 3), 0.16)
    valid = np.isfinite(scores)
    low = valid & (scores <= 0.5)
    high = valid & (scores > 0.5)
    colors[low] = bark + (middle - bark) * (scores[low, None] * 2)
    colors[high] = middle + (lichen - middle) * ((scores[high, None] - 0.5) * 2)
    colors[vegetation] = [0.20, 0.55, 0.27]
    return colors


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def write_tree_output(tree: TreeData, scores: np.ndarray, cv: dict) -> dict:
    import open3d as o3d
    from cluster_lichen_surface import colored_triangle_mesh

    output_dir = PILOT_ROOT / tree.uid / OUTPUT_VARIANT
    output_dir.mkdir(parents=True, exist_ok=True)
    vegetation = np.zeros(len(tree.labels), dtype=bool)
    for cluster, decision in tree.reviews.items():
        if decision == "vegetation":
            vegetation |= tree.labels == cluster
    valid = np.isfinite(scores)
    area = tree.areas_m2
    rows = {
        "scan_uid": tree.uid,
        "method": "balanced_pure_label_ridge_on_inverse_PCA_DINOv3_L_features",
        "source_variant": tree.source_dir.name,
        "observed_triangles": int((tree.labels >= 0).sum()),
        "material_triangles": int(valid.sum()),
        "vegetation_triangles": int(vegetation.sum()),
        "material_area_cm2": float(area[valid].sum() * 1e4),
        "estimated_lichen_area_cm2": float((area[valid] * scores[valid]).sum() * 1e4),
        "hard_bark_area_cm2": float(area[valid & (scores <= 0.35)].sum() * 1e4),
        "review_band_area_cm2": float(area[valid & (scores > 0.35) & (scores < 0.65)].sum() * 1e4),
        "hard_lichen_area_cm2": float(area[valid & (scores >= 0.65)].sum() * 1e4),
        "thresholds": {"bark_max": 0.35, "lichen_min": 0.65},
        "cross_validation": cv,
    }
    # NTFS is extremely slow for Open3D's many small PLY reads; parse on local APFS.
    with tempfile.TemporaryDirectory(prefix="que-lichen-material-") as temporary_dir:
        local_source = Path(temporary_dir) / "source.ply"
        local_output = Path(temporary_dir) / "material.ply"
        shutil.copyfile(tree.source_dir / "surface_fused_rgb_mesh.ply", local_source)
        source_mesh = o3d.io.read_triangle_mesh(str(local_source))
        vertices = np.asarray(source_mesh.vertices)
        triangles = np.asarray(source_mesh.triangles)
        if len(triangles) != len(scores):
            raise ValueError(f"{tree.uid}: mesh triangles do not match material scores")
        mesh = colored_triangle_mesh(vertices, triangles, probability_colors(scores, vegetation))
        if not o3d.io.write_triangle_mesh(str(local_output), mesh):
            raise OSError(f"failed to write {local_output}")
        temporary_mesh = output_dir / "material_probability_mesh.tmp.ply"
        shutil.copyfile(local_output, temporary_mesh)
        temporary_mesh.replace(output_dir / "material_probability_mesh.ply")
    temporary_npz = output_dir / "material_probabilities.tmp.npz"
    np.savez_compressed(temporary_npz, lichen_probability=scores, vegetation=vegetation,
                        labels=tree.labels, areas_m2=tree.areas_m2)
    temporary_npz.replace(output_dir / "material_probabilities.npz")
    atomic_json(output_dir / "material_probe_metrics.json", rows)
    return rows


def run(args: argparse.Namespace) -> dict:
    trees = load_trees(args.reviews)
    fold_rows = []
    for held_out in trees:
        if not any(decision in PURE_TARGETS for decision in held_out.reviews.values()):
            continue
        x, y, weights = training_rows(
            trees, args.sample_per_cluster, args.seed, excluded_uid=held_out.uid)
        coefficients = fit_probe(x, y, weights, args.ridge)
        fold_rows.extend(cluster_evaluation(held_out, predict_tree(held_out, coefficients)))
        print(f"validated {held_out.uid}", flush=True)
    cv = {
        "protocol": "leave_one_tree_out",
        "evaluated_trees": len({row["scan_uid"] for row in fold_rows}),
        "clusters": len(fold_rows),
        "mean_absolute_error": float(np.mean([row["absolute_error"] for row in fold_rows])),
        "directional_accuracy": float(np.mean([
            (row["predicted"] >= 0.5) == (row["target"] > 0.5)
            for row in fold_rows if row["target"] != 0.5
        ])),
        "rows": fold_rows,
    }
    x, y, weights = training_rows(trees, args.sample_per_cluster, args.seed)
    coefficients = fit_probe(x, y, weights, args.ridge)
    outputs = []
    for tree in trees:
        outputs.append(write_tree_output(tree, predict_tree(tree, coefficients), cv))
        print(f"wrote {tree.uid}", flush=True)
    model_dir = PILOT_ROOT / OUTPUT_VARIANT
    model_dir.mkdir(parents=True, exist_ok=True)
    temporary_model = model_dir / "material_probe_model.tmp.npz"
    np.savez_compressed(temporary_model, coefficients=coefficients)
    temporary_model.replace(model_dir / "material_probe_model.npz")
    summary = {
        "status": "pilot_not_ground_truth",
        "model": "DINOv3-L inverse-PCA feature + balanced pure-label ridge",
        "source_variants": list(SOURCE_VARIANTS),
        "trees": len(trees),
        "supervised_trees": sum(
            any(decision in PURE_TARGETS for decision in tree.reviews.values())
            for tree in trees
        ),
        "training_rows": len(x),
        "sample_per_cluster": args.sample_per_cluster,
        "ridge": args.ridge,
        "targets": PURE_TARGETS,
        "excluded_labels": ["mixed_bark", "mixed", "mixed_lichen", "vegetation", "uncertain"],
        "cross_validation": cv,
        "outputs": outputs,
    }
    atomic_json(model_dir / "material_probe_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviews", type=Path, default=REVIEWS_PATH)
    parser.add_argument("--sample-per-cluster", type=int, default=512)
    parser.add_argument("--ridge", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({
        "trees": result["trees"],
        "training_rows": result["training_rows"],
        "cv_mae": result["cross_validation"]["mean_absolute_error"],
        "cv_directional_accuracy": result["cross_validation"]["directional_accuracy"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
