import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import get_subject_splits, extract_subject_id

dummy_folders = [
    "BraTS-GLI-00001-000", "BraTS-GLI-00001-001",
    "BraTS-GLI-00002-000",
    "BraTS-GLI-00003-000", "BraTS-GLI-00003-001", "BraTS-GLI-00003-002",
    "BraTS-GLI-00004-000",
    "BraTS-GLI-00005-000", "BraTS-GLI-00005-001",
    "BraTS-GLI-00006-000",
    "BraTS-GLI-00007-000",
    "BraTS-GLI-00008-000",
    "BraTS-GLI-00009-000",
    "BraTS-GLI-00010-000",
]

train_f, val_f, test_f = get_subject_splits(dummy_folders, n_folds=3, fold=0, seed=42)

train_subs = set(extract_subject_id(f) for f in train_f)
val_subs   = set(extract_subject_id(f) for f in val_f)
test_subs  = set(extract_subject_id(f) for f in test_f)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

print(f"Total scans: {len(dummy_folders)}, Total unique subjects: {len(set(extract_subject_id(f) for f in dummy_folders))}")
print(f"Train subjects ({len(train_subs)}): {sorted(list(train_subs))}")
print(f"Val subjects   ({len(val_subs)}): {sorted(list(val_subs))}")
print(f"Test subjects  ({len(test_subs)}): {sorted(list(test_subs))}")
print("-" * 50)
print(f"train_subs ∩ val_subs  : {train_subs.intersection(val_subs)}")
print(f"train_subs ∩ test_subs : {train_subs.intersection(test_subs)}")
print(f"val_subs ∩ test_subs   : {val_subs.intersection(test_subs)}")
print("-" * 50)
assert len(train_subs.intersection(val_subs)) == 0, "Train-Val Leakage!"
assert len(train_subs.intersection(test_subs)) == 0, "Train-Test Leakage!"
assert len(val_subs.intersection(test_subs)) == 0, "Val-Test Leakage!"
print("STATUS: ZERO LEAKAGE CONFIRMED (All 3 pairwise intersections are empty sets).")
