import json
import os
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).parents[1]

# Sensitive data path - must be set via environment variable
UKBB_METADATA_CSV = os.getenv("UKBB_METADATA_CSV")
assert UKBB_METADATA_CSV is not None, "Path for UKBB_METADATA_CSV must be set"
UKBB_METADATA_CSV_PATH = Path(UKBB_METADATA_CSV)

UKBB_ROOT = ROOT / "data/sourcedata"

# Total number of UKBB subjects with imaging data
UKBB_NUM_SUBJECTS = 740

# Number of batches for stratified random splits
NUM_BATCHES = 20
SEED = 42


def main():
    outpath = ROOT / "metadata/ukbb_subject_batch_splits.json"
    assert not outpath.exists(), f"output splits {outpath} already exist"

    # Load metadata
    df = pd.read_csv(UKBB_METADATA_CSV_PATH, dtype={"participant.eid": str})
    df = df.rename(columns={
        "participant.eid": "eid",
        "participant.p31": "sex",
        "participant.p21003_i0": "age"
    })

    # Get subjects with imaging data from .bold.npy files
    all_subjects = np.array(sorted([
        p.name.split("_")[0].replace("sub-", "")
        for p in UKBB_ROOT.glob("sub-*_ses-*_mod-rfMRI.bold.npy")
    ]))
    print(f"Found {len(all_subjects)} subjects with imaging data")
    print(f"First 10 subjects: {all_subjects[:10]}")
    assert len(all_subjects) == UKBB_NUM_SUBJECTS, f"Expected {UKBB_NUM_SUBJECTS} subjects, found {len(all_subjects)}"

    # Filter metadata to matched subjects
    df = df[df["eid"].isin(all_subjects)].copy()
    df = df.set_index("eid").loc[all_subjects].reset_index()

    print(f"\nMatched {len(df)} subjects in metadata")
    print(f"Age range: {df['age'].min():.1f} - {df['age'].max():.1f} years")
    print(f"Sex distribution: {df['sex'].value_counts().to_dict()}")

    # Create quantile-based age bins (3 bins)
    df["age_bin"] = pd.qcut(df["age"], q=3, labels=False)

    print(f"\nAge bins:")
    for bin_idx in range(3):
        bin_df = df[df["age_bin"] == bin_idx]
        print(f"  Bin {bin_idx}: {bin_df['age'].min():.1f}-{bin_df['age'].max():.1f} years ({len(bin_df)} subjects)")

    # Create stratification groups: sex × age_bin (6 groups)
    df["strata"] = df["sex"].astype(str) + "_" + df["age_bin"].astype(str)
    strata_labels = df["strata"].values

    print(f"\nStratification groups:")
    for strata in sorted(df["strata"].unique()):
        count = (df["strata"] == strata).sum()
        print(f"  {strata}: {count} subjects")

    # Use StratifiedKFold to create balanced splits
    splitter = StratifiedKFold(n_splits=NUM_BATCHES, shuffle=True, random_state=SEED)

    splits = {}
    for ii, (_, ind) in enumerate(splitter.split(all_subjects, y=strata_labels)):
        batch_subjects = all_subjects[ind].tolist()
        splits[f"batch-{ii:02d}"] = batch_subjects
        print(f"batch-{ii:02d}: {len(batch_subjects)} subjects")

    # Verify train/val/test split counts
    train_subs = [s for i in range(14) for s in splits[f"batch-{i:02d}"]]
    val_subs = [s for i in range(14, 17) for s in splits[f"batch-{i:02d}"]]
    test_subs = [s for i in range(17, 20) for s in splits[f"batch-{i:02d}"]]

    print(f"\nSplit counts:")
    print(f"  Train (batches 0-13): {len(train_subs)} subjects")
    print(f"  Val (batches 14-16): {len(val_subs)} subjects")
    print(f"  Test (batches 17-19): {len(test_subs)} subjects")

    outpath.parent.mkdir(exist_ok=True, parents=True)
    with outpath.open("w") as f:
        json.dump(splits, f, indent=2)

    print(f"\nSaved splits to {outpath}")


if __name__ == "__main__":
    main()
