import sys, os, time
sys.path.insert(0, os.path.abspath('.'))

import torch
from torch.optim import Adam

DATASET_PATH     = os.path.join(os.path.expanduser("~"), "Brain_Tumor_Segmentation",
                    "ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData")
QUICK_TEST       = False
RESUME           = True
USE_CHANNEL_ATTN = True
USE_SPATIAL_ATTN = True
MODEL_NAME       = "attention_unet3d_full"

from src.config import (BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
    MAX_PATIENTS, N_FOLDS, DEVICE, RANDOM_SEED, EARLY_STOP_PATIENCE, LR_PATIENCE, LR_FACTOR)
from src.models  import AttentionUNet3D
from src.losses  import CombinedLoss
from src.dataset import get_dataloaders
from src.utils   import (set_seed, save_checkpoint, load_checkpoint,
                         train_one_epoch, validate_one_epoch,
                         get_scheduler, get_writer, format_time)

set_seed(RANDOM_SEED)
print(f"Device  : {DEVICE}")
print(f"Model   : {MODEL_NAME}")

train_loader, val_loader, test_loader, _ = get_dataloaders(
    dataset_path=DATASET_PATH, batch_size=BATCH_SIZE, num_workers=3,
    fold=0, n_folds=N_FOLDS, max_patients=MAX_PATIENTS,
    quick_test=QUICK_TEST, quick_test_n=10,
)
print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

model = AttentionUNet3D(in_channels=4, out_channels=4, init_features=32,
    use_channel_attn=USE_CHANNEL_ATTN, use_spatial_attn=USE_SPATIAL_ATTN).to(DEVICE)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
criterion = CombinedLoss(num_classes=4, dice_weight=0.5, focal_weight=0.5)
scheduler = get_scheduler(optimizer, patience=LR_PATIENCE, factor=LR_FACTOR)
writer    = get_writer(MODEL_NAME)

start_epoch, best_val_dice = 0, 0.0
if RESUME:
    start_epoch, best_val_dice = load_checkpoint(model, optimizer, model_name=MODEL_NAME, prefer_best=False)

max_epochs = NUM_EPOCHS
no_improve_count = 0
t0 = time.time()

for epoch in range(start_epoch, max_epochs):
    ep_start = time.time()
    train_m = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE, epoch, writer)
    val_m   = validate_one_epoch(model, val_loader, criterion, DEVICE, epoch, writer)
    scheduler.step(val_m['wt_dice'])
    is_best = val_m['wt_dice'] > best_val_dice
    if is_best:
        best_val_dice = val_m['wt_dice']
        no_improve_count = 0
    else:
        no_improve_count += 1
    save_checkpoint(model, optimizer, epoch, val_m['wt_dice'], model_name=MODEL_NAME, is_best=is_best)
    eta = (time.time()-ep_start)*(max_epochs-epoch-1)
    print(f"Epoch [{epoch+1:3d}/{max_epochs}] | Train: {train_m['loss']:.4f} | Val: {val_m['loss']:.4f} | WT Dice: {val_m['wt_dice']:.4f} | Best: {best_val_dice:.4f} | ETA: {format_time(eta)}")
    sys.stdout.flush()
    if no_improve_count >= EARLY_STOP_PATIENCE:
        print(f"Early stopping at epoch {epoch+1}")
        break

writer.close()
print(f"\n🏁 Done! Best WT Dice: {best_val_dice:.4f} | Total: {format_time(time.time()-t0)}")
