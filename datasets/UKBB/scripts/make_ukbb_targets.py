import json
import os
from pathlib import Path

import pandas as pd
import numpy as np

ROOT = Path(__file__).parents[1]

# Sensitive data path - must be set via environment variable
UKBB_METADATA_CSV = os.getenv("UKBB_METADATA_CSV")
assert UKBB_METADATA_CSV is not None, "Path for UKBB_METADATA_CSV must be set"
UKBB_METADATA_CSV_PATH = Path(UKBB_METADATA_CSV)

UKBB_ROOT = ROOT / "data/sourcedata"

# Total number of UKBB subjects with imaging data
UKBB_NUM_SUBJECTS = 740


def main():
    # Load metadata
    df = pd.read_csv(UKBB_METADATA_CSV_PATH, dtype={"participant.eid": str})
    df = df.rename(columns={
        "participant.eid": "Subject",
        "participant.p31": "sex",
        "participant.p21003_i0": "age"
    })

    # Get subjects with imaging data from .bold.npy files
    all_subjects = sorted([
        p.name.split("_")[0].replace("sub-", "")
        for p in UKBB_ROOT.glob("sub-*_ses-*_mod-rfMRI.bold.npy")
    ])
    print(f"Found {len(all_subjects)} subjects with imaging data")
    assert len(all_subjects) == UKBB_NUM_SUBJECTS, f"Expected {UKBB_NUM_SUBJECTS} subjects"

    # Filter metadata to matched subjects
    df = df[df["Subject"].isin(all_subjects)].copy()
    df = df.set_index("Subject").loc[all_subjects].reset_index()

    print(f"\nMatched {len(df)} subjects in metadata")
    print(f"Age range: {df['age'].min():.1f} - {df['age'].max():.1f} years")
    print(f"Sex distribution: {df['sex'].value_counts().to_dict()}")

    # Create quantile-based age bins (3 bins)
    df["Age_Q"] = pd.qcut(df["age"], q=3, labels=False)
    age_bins = pd.qcut(df["age"], q=3, retbins=True)[1]

    print(f"\nAge bins:")
    for bin_idx in range(3):
        bin_df = df[df["Age_Q"] == bin_idx]
        print(f"  Bin {bin_idx}: {bin_df['age'].min():.1f}-{bin_df['age'].max():.1f} years ({len(bin_df)} subjects)")

    # Map sex: 0=Female, 1=Male
    df["Gender"] = df["sex"].map({0: "F", 1: "M"})

    # Rename columns for output
    output_df = df[["Subject", "age", "Age_Q", "Gender"]].copy()
    output_df = output_df.rename(columns={"age": "Age"})

    # Save main phenotypic targets CSV
    pheno_csv_path = ROOT / "metadata/ukbb_pheno_targets.csv"
    pheno_csv_path.parent.mkdir(exist_ok=True, parents=True)
    output_df.to_csv(pheno_csv_path, index=False)
    print(f"\nSaved phenotypic targets to {pheno_csv_path}")

    # Create targets directory
    targets_dir = ROOT / "metadata/targets"
    targets_dir.mkdir(exist_ok=True, parents=True)

    # Generate Age target files
    age_bins_list = age_bins[1:-1].tolist()  # Inner boundaries only
    age_label_counts = df["Age_Q"].value_counts().sort_index().tolist()

    age_target_info = {
        "target": "Age",
        "na_count": 0,
        "bins": [int(b) for b in age_bins_list],
        "label_counts": age_label_counts
    }

    age_target_map = {
        row["Subject"]: int(row["Age_Q"])
        for _, row in output_df.iterrows()
    }

    # Save Age target files
    with (targets_dir / "ukbb_target_info_Age.json").open("w") as f:
        json.dump(age_target_info, f, indent=2)

    with (targets_dir / "ukbb_target_map_Age.json").open("w") as f:
        json.dump(age_target_map, f, indent=2)

    print(f"Saved Age target info to {targets_dir / 'ukbb_target_info_Age.json'}")
    print(f"Saved Age target map to {targets_dir / 'ukbb_target_map_Age.json'}")

    # Generate Gender target files
    gender_counts = df["Gender"].value_counts().to_dict()
    gender_label_counts = [gender_counts.get("F", 0), gender_counts.get("M", 0)]

    gender_target_info = {
        "target": "Gender",
        "na_count": 0,
        "bins": ["F", "M"],
        "label_counts": gender_label_counts
    }

    gender_target_map = {
        row["Subject"]: row["Gender"]
        for _, row in output_df.iterrows()
    }

    # Save Gender target files
    with (targets_dir / "ukbb_target_info_Gender.json").open("w") as f:
        json.dump(gender_target_info, f, indent=2)

    with (targets_dir / "ukbb_target_map_Gender.json").open("w") as f:
        json.dump(gender_target_map, f, indent=2)

    print(f"Saved Gender target info to {targets_dir / 'ukbb_target_info_Gender.json'}")
    print(f"Saved Gender target map to {targets_dir / 'ukbb_target_map_Gender.json'}")

    # Print summary
    print("\n=== Summary ===")
    print(f"Total subjects: {len(output_df)}")
    print(f"Age bins: {age_bins_list} ({age_label_counts})")
    print(f"Gender distribution: F={gender_label_counts[0]}, M={gender_label_counts[1]}")


if __name__ == "__main__":
    main()
