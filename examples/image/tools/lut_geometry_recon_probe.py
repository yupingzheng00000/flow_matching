import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from flow_matching.examples.image.tools.lut_geometry_inspect import (
    load_lut_from_checkpoint as load_lut_from_checkpoint_inspect,
)
from flow_matching.examples.image.training.train_loop import (
    _compute_embedding_collapse_metrics,
)

try:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    _HAS_SKLEARN = True
except Exception:
    PCA = None  # type: ignore[assignment]
    TSNE = None  # type: ignore[assignment]
    _HAS_SKLEARN = False


def _load_flat_embeddings(
    checkpoint_path: str,
    lut_key: str,
) -> Tuple[torch.Tensor, Dict[str, int]]:
    weight, metadata = load_lut_from_checkpoint_inspect(Path(checkpoint_path), lut_key=lut_key)
    if weight.ndim == 2:
        weight = weight.unsqueeze(-1)
    if weight.ndim != 3:
        raise ValueError(f"Expected LUT weight with 2 or 3 dims; got shape {tuple(weight.shape)}")
    channels, vocab_size, embed_dim = weight.shape
    flat = weight.reshape(channels * vocab_size, embed_dim).to(dtype=torch.float32)
    return flat, metadata


def _compute_singular_values(flat_embeddings: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        centered = flat_embeddings - flat_embeddings.mean(dim=0, keepdim=True)
        gram = centered.transpose(0, 1) @ centered
        eigenvalues = torch.linalg.eigvalsh(gram).clamp_min(0.0)
        sigma = torch.sqrt(eigenvalues)
        sigma_sorted, _ = torch.sort(sigma, descending=True)
    return sigma_sorted.cpu().numpy()


def _save_metrics_json(
    output_path: Path,
    metrics: Dict[str, float],
    metadata: Dict[str, int],
) -> None:
    def _convert_meta_value(value):
        # Preserve non-numeric metadata (e.g., embed_range="pm1") as-is.
        try:
            # Torch / NumPy integer-like values are converted to plain int
            if isinstance(value, (int,)):
                return int(value)
            return value
        except Exception:
            return value

    payload = {
        "metrics": metrics,
        "meta": {name: _convert_meta_value(value) for name, value in metadata.items()},
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _sample_rows(flat_embeddings: torch.Tensor, max_rows: int) -> np.ndarray:
    total_rows = flat_embeddings.shape[0]
    if total_rows <= max_rows:
        return flat_embeddings.cpu().numpy()
    indices = torch.randperm(total_rows)[:max_rows]
    return flat_embeddings[indices].cpu().numpy()


def _plot_singular_values(
    sigma_first: np.ndarray,
    sigma_second: Optional[np.ndarray],
    label_first: str,
    label_second: Optional[str],
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(6, 4))
    index_first = np.arange(1, sigma_first.shape[0] + 1)
    axis.plot(index_first, sigma_first, marker="o", label=label_first)
    if sigma_second is not None and label_second is not None:
        index_second = np.arange(1, sigma_second.shape[0] + 1)
        axis.plot(index_second, sigma_second, marker="o", label=label_second)
    axis.set_yscale("log")
    axis.set_xlabel("Singular value index")
    axis.set_ylabel("σ")
    axis.set_title("LUT singular value spectrum")
    axis.legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def _plot_pca_scatter(
    flat_first: torch.Tensor,
    flat_second: Optional[torch.Tensor],
    label_first: str,
    label_second: Optional[str],
    max_codes: int,
    output_path: Path,
) -> None:
    if PCA is None:
        return
    first_subset = _sample_rows(flat_first, max_codes)
    if first_subset.shape[0] < 2:
        return
    if flat_second is not None:
        second_subset = _sample_rows(flat_second, max_codes)
        if second_subset.shape[0] < 2:
            return
        combined = np.concatenate([first_subset, second_subset], axis=0)
        projector = PCA(n_components=2)
        combined_coords = projector.fit_transform(combined)
        count_first = first_subset.shape[0]
        coords_first = combined_coords[:count_first]
        coords_second = combined_coords[count_first:]
    else:
        projector = PCA(n_components=2)
        coords_first = projector.fit_transform(first_subset)
        coords_second = None
    figure, axis = plt.subplots(figsize=(6, 5))
    axis.scatter(coords_first[:, 0], coords_first[:, 1], s=6, alpha=0.6, label=label_first)
    if coords_second is not None and label_second is not None:
        axis.scatter(
            coords_second[:, 0],
            coords_second[:, 1],
            s=6,
            alpha=0.6,
            label=label_second,
        )
    axis.set_xlabel("PC1")
    axis.set_ylabel("PC2")
    axis.set_title("LUT PCA projection")
    axis.legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def _plot_tsne_scatter(
    flat_first: torch.Tensor,
    flat_second: Optional[torch.Tensor],
    label_first: str,
    label_second: Optional[str],
    max_codes: int,
    output_path: Path,
) -> None:
    if not _HAS_SKLEARN or TSNE is None:
        return
    first_subset = _sample_rows(flat_first, max_codes)
    if first_subset.shape[0] < 2:
        return
    if flat_second is not None:
        second_subset = _sample_rows(flat_second, max_codes)
        if second_subset.shape[0] < 2:
            return
        combined = np.concatenate([first_subset, second_subset], axis=0)
        count_first = first_subset.shape[0]
    else:
        combined = first_subset
        count_first = combined.shape[0]
    perplexity = min(30.0, max(5.0, (combined.shape[0] - 1) / 3.0))
    projector = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate="auto",
        init="random",
        random_state=0,
    )
    combined_coords = projector.fit_transform(combined)
    coords_first = combined_coords[:count_first]
    coords_second = combined_coords[count_first:] if flat_second is not None else None
    figure, axis = plt.subplots(figsize=(6, 5))
    axis.scatter(coords_first[:, 0], coords_first[:, 1], s=6, alpha=0.6, label=label_first)
    if coords_second is not None and label_second is not None:
        axis.scatter(
            coords_second[:, 0],
            coords_second[:, 1],
            s=6,
            alpha=0.6,
            label=label_second,
        )
    axis.set_xlabel("t-SNE dim 1")
    axis.set_ylabel("t-SNE dim 2")
    axis.set_title("LUT t-SNE projection")
    axis.legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def _plot_similarity_heatmaps(
    flat_first: torch.Tensor,
    flat_second: Optional[torch.Tensor],
    label_first: str,
    label_second: Optional[str],
    subset_size: int,
    output_path: Path,
) -> None:
    similarity_matrices = []
    titles = []
    first_subset = _sample_rows(flat_first, subset_size)
    if first_subset.shape[0] >= 2:
        norms_first = np.linalg.norm(first_subset, axis=1, keepdims=True).clip(min=1e-12)
        normalized_first = first_subset / norms_first
        similarity_first = normalized_first @ normalized_first.T
        similarity_matrices.append(similarity_first)
        titles.append(label_first)
    if flat_second is not None:
        second_subset = _sample_rows(flat_second, subset_size)
        if second_subset.shape[0] >= 2:
            norms_second = np.linalg.norm(second_subset, axis=1, keepdims=True).clip(min=1e-12)
            normalized_second = second_subset / norms_second
            similarity_second = normalized_second @ normalized_second.T
            similarity_matrices.append(similarity_second)
            titles.append(label_second)
    if not similarity_matrices:
        return
    column_count = len(similarity_matrices)
    figure, axes = plt.subplots(1, column_count, figsize=(6 * column_count, 5))
    if column_count == 1:
        axes = [axes]
    for axis, matrix, title in zip(axes, similarity_matrices, titles):
        sns.heatmap(matrix, ax=axis, cmap="viridis", vmin=-1.0, vmax=1.0)
        axis.set_title(f"Similarity ({title})")
        axis.set_xlabel("Index")
        axis.set_ylabel("Index")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def _plot_index0_distance_curves(
    flat_first: torch.Tensor,
    flat_second: Optional[torch.Tensor],
    label_first: str,
    label_second: Optional[str],
    reference_index: int,
    output_path: Path,
) -> None:
    distance_curves = []
    labels = []
    for embeddings, label in ((flat_first, label_first), (flat_second, label_second)):
        if embeddings is None:
            continue
        total_rows = embeddings.shape[0]
        if total_rows == 0:
            continue
        index = int(max(0, min(reference_index, total_rows - 1)))
        with torch.no_grad():
            reference = embeddings[index : index + 1]
            distances = torch.cdist(reference, embeddings)[0].cpu().numpy()
        distances_sorted = np.sort(distances)
        distance_curves.append(distances_sorted)
        labels.append(label)
    if not distance_curves:
        return
    figure, axis = plt.subplots(figsize=(6, 4))
    for distances_sorted, label in zip(distance_curves, labels):
        ranks = np.arange(distances_sorted.shape[0])
        axis.plot(ranks, distances_sorted, label=label)
    axis.set_xlabel("Rank (sorted by distance)")
    axis.set_ylabel("Distance")
    axis.set_title(f"Distance profile (reference index={reference_index})")
    axis.legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def _analyze_checkpoints(
    checkpoint_first: str,
    checkpoint_second: Optional[str],
    args: argparse.Namespace,
) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    lut_key = args.lut_key
    flat_first, meta_first = _load_flat_embeddings(checkpoint_first, lut_key=lut_key)
    metrics_first = _compute_embedding_collapse_metrics(flat_first)
    sigma_first = _compute_singular_values(flat_first)
    first_tag = args.label_first
    _save_metrics_json(output_dir / f"{first_tag}_metrics.json", metrics_first, meta_first)
    if checkpoint_second is not None:
        flat_second, meta_second = _load_flat_embeddings(checkpoint_second, lut_key=lut_key)
        metrics_second = _compute_embedding_collapse_metrics(flat_second)
        sigma_second = _compute_singular_values(flat_second)
        second_tag = args.label_second
        _save_metrics_json(output_dir / f"{second_tag}_metrics.json", metrics_second, meta_second)
    else:
        flat_second = None
        sigma_second = None
        second_tag = None
    _plot_singular_values(
        sigma_first=sigma_first,
        sigma_second=sigma_second,
        label_first=first_tag,
        label_second=second_tag,
        output_path=output_dir / "singular_values.png",
    )
    _plot_pca_scatter(
        flat_first=flat_first,
        flat_second=flat_second,
        label_first=first_tag,
        label_second=second_tag,
        max_codes=args.num_pca_codes,
        output_path=output_dir / "pca_scatter.png",
    )
    if not args.no_tsne:
        _plot_tsne_scatter(
            flat_first=flat_first,
            flat_second=flat_second,
            label_first=first_tag,
            label_second=second_tag,
            max_codes=args.num_tsne_codes,
            output_path=output_dir / "tsne_scatter.png",
        )
    _plot_similarity_heatmaps(
        flat_first=flat_first,
        flat_second=flat_second,
        label_first=first_tag,
        label_second=second_tag,
        subset_size=args.similarity_k,
        output_path=output_dir / "similarity_heatmap.png",
    )
    _plot_index0_distance_curves(
        flat_first=flat_first,
        flat_second=flat_second,
        label_first=first_tag,
        label_second=second_tag,
        reference_index=args.ref_index,
        output_path=output_dir / "index0_distance_curve.png",
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="LUT geometry probe for reconstruction ablations.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Single checkpoint to analyze.",
    )
    parser.add_argument(
        "--checkpoint_no_recon",
        type=str,
        default=None,
        help="Baseline checkpoint with lut_recon_weight=0.",
    )
    parser.add_argument(
        "--checkpoint_recon",
        type=str,
        default=None,
        help="Checkpoint with lut_recon_weight>0.",
    )
    parser.add_argument(
        "--lut_key",
        type=str,
        default="metric_learnable_lut",
        help="Key used in checkpoint['extra_modules'] for LUT (legacy checkpoints).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save metrics JSON and figures.",
    )
    parser.add_argument(
        "--label_first",
        type=str,
        default="no_recon",
        help="Label for the first checkpoint in plots.",
    )
    parser.add_argument(
        "--label_second",
        type=str,
        default="recon",
        help="Label for the second checkpoint in plots.",
    )
    parser.add_argument(
        "--num_pca_codes",
        type=int,
        default=512,
        help="Maximum number of LUT codes used for PCA visualization per checkpoint.",
    )
    parser.add_argument(
        "--num_tsne_codes",
        type=int,
        default=400,
        help="Maximum number of LUT codes used for t-SNE visualization per checkpoint.",
    )
    parser.add_argument(
        "--similarity_k",
        type=int,
        default=64,
        help="Number of codes to sample when building similarity heatmaps.",
    )
    parser.add_argument(
        "--ref_index",
        type=int,
        default=0,
        help="Reference code index for distance profile curves.",
    )
    parser.add_argument(
        "--no_tsne",
        action="store_true",
        help="Disable t-SNE visualization even when sklearn is available.",
    )
    return parser


def main(argv: Optional[Tuple[str, ...]] = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    single_checkpoint = args.checkpoint
    pair_first = args.checkpoint_no_recon
    pair_second = args.checkpoint_recon
    if single_checkpoint is not None and (pair_first is not None or pair_second is not None):
        raise SystemExit("Specify either --checkpoint or the pair (--checkpoint_no_recon and --checkpoint_recon), not both.")
    if single_checkpoint is not None:
        _analyze_checkpoints(single_checkpoint, None, args)
    else:
        if pair_first is None or pair_second is None:
            raise SystemExit("Either --checkpoint must be set, or both --checkpoint_no_recon and --checkpoint_recon must be provided.")
        _analyze_checkpoints(pair_first, pair_second, args)


if __name__ == "__main__":
    main()
