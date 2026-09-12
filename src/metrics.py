"""
metrics.py — 15-Metric Evaluation Suite for 3D Brain Tumor Segmentation
Computes 5 clinical metrics across 3 BraTS sub-regions:
  - Sub-regions : WT (Whole Tumor), TC (Tumor Core), ET (Enhancing Tumor)
  - Metrics     : Dice Score, IoU (Jaccard), HD95 (mm), Sensitivity, Specificity
"""

from typing import Dict, List, Tuple, Optional, Union
import numpy as np
from scipy.ndimage import binary_erosion
from scipy.spatial import cKDTree


# =============================================================================
# 1. Clinical Sub-Region Extraction
# =============================================================================

def get_regions(seg: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Extracts 3 standard BraTS clinical sub-region binary masks from multi-class label map:
      - WT (Whole Tumor)   : Labels 1 + 2 + 3 (Necrotic Core + Edema + Enhancing)
      - TC (Tumor Core)    : Labels 1 + 3     (Necrotic Core + Enhancing)
      - ET (Enhancing)     : Label 3          (Active Enhancing Rim)

    Args:
        seg : (H, W, D) integer array with labels in {0, 1, 2, 3}
    Returns:
        Dict mapping region key ('WT', 'TC', 'ET') to boolean array (H, W, D)
    """
    return {
        "WT": (seg > 0),
        "TC": ((seg == 1) | (seg == 3)),
        "ET": (seg == 3),
    }


# =============================================================================
# 2. Metric Functions (Binary 3D Masks)
# =============================================================================

def dice_score(pred: np.ndarray, gt: np.ndarray, smooth: float = 1e-6) -> float:
    """Dice Similarity Coefficient (DSC) = 2|P ∩ G| / (|P| + |G|)."""
    p = pred.astype(bool)
    g = gt.astype(bool)
    intersection = (p & g).sum()
    cardinality  = p.sum() + g.sum()
    if cardinality == 0:
        return 1.0  # Both empty = perfect match
    return float((2.0 * intersection + smooth) / (cardinality + smooth))


def iou_score(pred: np.ndarray, gt: np.ndarray, smooth: float = 1e-6) -> float:
    """Intersection over Union (Jaccard Index) = |P ∩ G| / |P ∪ G|."""
    p = pred.astype(bool)
    g = gt.astype(bool)
    intersection = (p & g).sum()
    union        = (p | g).sum()
    if union == 0:
        return 1.0
    return float((intersection + smooth) / (union + smooth))


def sensitivity(pred: np.ndarray, gt: np.ndarray) -> float:
    """Sensitivity (True Positive Rate / Recall) = TP / (TP + FN)."""
    p = pred.astype(bool)
    g = gt.astype(bool)
    tp = (p & g).sum()
    fn = (~p & g).sum()
    if (tp + fn) == 0:
        return 1.0 if p.sum() == 0 else 0.0
    return float(tp / (tp + fn + 1e-8))


def specificity(pred: np.ndarray, gt: np.ndarray) -> float:
    """Specificity (True Negative Rate) = TN / (TN + FP)."""
    p = pred.astype(bool)
    g = gt.astype(bool)
    tn = (~p & ~g).sum()
    fp = (p & ~g).sum()
    if (tn + fp) == 0:
        return 1.0
    return float(tn / (tn + fp + 1e-8))


def hausdorff_distance_95(pred: np.ndarray, gt: np.ndarray,
                          voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)) -> float:
    """
    95th percentile Hausdorff Distance (HD95) in physical millimetres (mm).
    Robust to empty predictions / ground truths without throwing exceptions.

    Returns:
        HD95 value in mm, or np.nan if either mask contains no boundary surface.
    """
    p = pred.astype(bool)
    g = gt.astype(bool)

    if p.sum() == 0 and g.sum() == 0:
        return 0.0  # Both empty
    if p.sum() == 0 or g.sum() == 0:
        return np.nan  # Mismatch with completely missing mask

    def extract_surface_voxels(mask: np.ndarray) -> np.ndarray:
        eroded = binary_erosion(mask)
        surface_mask = mask & (~eroded)
        coords = np.argwhere(surface_mask)
        if len(coords) == 0:
            # Fallback if structure is a single isolated voxel
            coords = np.argwhere(mask)
        return coords.astype(np.float64)

    pred_pts = extract_surface_voxels(p)
    gt_pts   = extract_surface_voxels(g)

    if len(pred_pts) == 0 or len(gt_pts) == 0:
        return np.nan

    # Scale coordinates by physical voxel dimensions (e.g. 1x1x1 mm)
    spacing_arr = np.array(voxel_spacing, dtype=np.float64)
    pred_pts *= spacing_arr
    gt_pts   *= spacing_arr

    try:
        tree_gt   = cKDTree(gt_pts)
        tree_pred = cKDTree(pred_pts)

        d_pred_to_gt, _ = tree_gt.query(pred_pts, k=1)
        d_gt_to_pred, _ = tree_pred.query(gt_pts, k=1)

        all_distances = np.concatenate([d_pred_to_gt, d_gt_to_pred])
        return float(np.percentile(all_distances, 95))
    except Exception:
        return np.nan


# =============================================================================
# 3. Patient-Level & Test-Set Aggregation
# =============================================================================

def compute_patient_metrics(pred_seg: np.ndarray,
                            gt_seg: np.ndarray,
                            voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
                            ) -> Dict[str, Dict[str, float]]:
    """
    Computes all 5 metrics across all 3 BraTS sub-regions (WT, TC, ET) for one patient.

    Returns:
        nested dict: {region: {'dice': float, 'iou': float, 'hd95': float,
                               'sensitivity': float, 'specificity': float}}
    """
    pred_regions = get_regions(pred_seg)
    gt_regions   = get_regions(gt_seg)

    results: Dict[str, Dict[str, float]] = {}
    for region in ["WT", "TC", "ET"]:
        p = pred_regions[region]
        g = gt_regions[region]
        results[region] = {
            "dice":        dice_score(p, g),
            "iou":         iou_score(p, g),
            "hd95":        hausdorff_distance_95(p, g, voxel_spacing),
            "sensitivity": sensitivity(p, g),
            "specificity": specificity(p, g),
        }
    return results


def aggregate_metrics(all_results: List[Dict[str, Dict[str, float]]]
                      ) -> Dict[str, Dict[str, Dict[str, Union[float, List[float]]]]]:
    """
    Aggregates per-patient evaluation results across the test cohort.
    Gracefully handles empty/NaN values for HD95.

    Returns:
        {region: {metric: {'mean': float, 'std': float, 'median': float, 'raw': List[float]}}}
    """
    aggregated: Dict[str, Dict[str, Dict[str, Union[float, List[float]]]]] = {}

    for region in ["WT", "TC", "ET"]:
        aggregated[region] = {}
        for metric in ["dice", "iou", "hd95", "sensitivity", "specificity"]:
            vals = [r[region][metric] for r in all_results if metric in r[region]]
            clean_vals = [float(v) for v in vals if not np.isnan(v)]

            if clean_vals:
                aggregated[region][metric] = {
                    "mean":   float(np.mean(clean_vals)),
                    "std":    float(np.std(clean_vals)),
                    "median": float(np.median(clean_vals)),
                    "raw":    clean_vals,
                }
            else:
                aggregated[region][metric] = {
                    "mean": np.nan, "std": np.nan, "median": np.nan, "raw": []
                }

    return aggregated


def print_metrics_table(aggregated: Dict, model_name: str = "Model") -> None:
    """Prints publication-ready formatted ASCII table of all 15 metrics."""
    print(f"\n{'='*68}")
    print(f"  Evaluation Results: {model_name}")
    print(f"{'='*68}")
    print(f"  {'Metric':<20} {'WT (Whole)':>14} {'TC (Core)':>14} {'ET (Enhancing)':>14}")
    print(f"  {'-'*64}")

    metrics_list = [
        ("Dice Similarity ↑", "dice", "{:.4f}±{:.3f}"),
        ("IoU (Jaccard) ↑",   "iou",  "{:.4f}±{:.3f}"),
        ("HD95 ↓ (mm)",       "hd95", "{:.2f}±{:.2f}"),
        ("Sensitivity ↑",     "sensitivity", "{:.4f}±{:.3f}"),
        ("Specificity ↑",     "specificity", "{:.4f}±{:.3f}"),
    ]

    for label, key, fmt in metrics_list:
        row = f"  {label:<20}"
        for region in ["WT", "TC", "ET"]:
            m = aggregated[region][key]
            mean, std = m["mean"], m["std"]
            if np.isnan(mean):
                row += f" {'N/A':>14}"
            else:
                val_str = fmt.format(mean, std)
                row += f" {val_str:>14}"
        print(row)
    print(f"{'='*68}\n")
