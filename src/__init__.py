"""
src — 3D Brain Tumor Segmentation Package
"""

from src.config import (
    DATASET_PATH,
    VALIDATION_PATH,
    PREPROCESSED_PATH,
    CHECKPOINT_DIR,
    RESULTS_DIR,
    LOGS_DIR,
    MODALITIES,
    LABEL_NAMES,
    REGION_LABELS,
    NUM_CLASSES,
    PATCH_SIZE,
    PATCH_OVERLAP,
    RANDOM_SEED,
    DEVICE,
    get_device,
    ensure_directories,
)

from src.dataset import (
    get_patient_folders,
    get_file_paths,
    load_volume,
    normalize_volume,
    remap_labels,
    remap_labels_brats2023,
    load_patient,
    extract_subject_id,
    get_subject_splits,
    extract_random_patch,
    augment_patch,
    BraTS2023Dataset,
    BraTS2023InferenceDataset,
    get_dataloaders,
)

from src.models import (
    UNet3D,
    AttentionUNet3D,
    SEBlock3D,
    SpatialAttentionGate3D,
    DoubleConv3D,
    EncoderBlock,
    AttentionDecoderBlock,
    build_model,
    count_parameters,
)

from src.losses import (
    DiceLoss,
    FocalLoss,
    CombinedLoss,
    compute_class_weights_from_train_split,
)

from src.metrics import (
    get_regions,
    dice_score,
    iou_score,
    sensitivity,
    specificity,
    hausdorff_distance_95,
    compute_patient_metrics,
    aggregate_metrics,
    print_metrics_table,
)

from src.utils import (
    set_seed,
    save_checkpoint,
    load_checkpoint,
    sliding_window_inference,
    train_one_epoch,
    validate_full_volume,
    get_scheduler,
    get_writer,
    format_time,
)

__all__ = [
    # Config
    "DATASET_PATH", "VALIDATION_PATH", "PREPROCESSED_PATH", "CHECKPOINT_DIR",
    "RESULTS_DIR", "LOGS_DIR", "MODALITIES", "LABEL_NAMES", "REGION_LABELS",
    "NUM_CLASSES", "PATCH_SIZE", "PATCH_OVERLAP", "RANDOM_SEED", "DEVICE",
    "get_device", "ensure_directories",
    # Dataset
    "get_patient_folders", "get_file_paths", "load_volume", "normalize_volume",
    "remap_labels", "load_patient", "extract_subject_id", "get_subject_splits",
    "extract_random_patch", "augment_patch", "BraTS2023Dataset",
    "BraTS2023InferenceDataset", "get_dataloaders",
    # Models
    "UNet3D", "AttentionUNet3D", "SEBlock3D", "SpatialAttentionGate3D",
    "DoubleConv3D", "EncoderBlock", "AttentionDecoderBlock", "build_model", "count_parameters",
    # Losses
    "DiceLoss", "FocalLoss", "CombinedLoss", "compute_class_weights_from_train_split",
    # Metrics
    "get_regions", "dice_score", "iou_score", "sensitivity", "specificity",
    "hausdorff_distance_95", "compute_patient_metrics", "aggregate_metrics", "print_metrics_table",
    # Utils
    "set_seed", "save_checkpoint", "load_checkpoint", "sliding_window_inference",
    "train_one_epoch", "validate_full_volume", "get_scheduler", "get_writer", "format_time"
]
