import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parents[1]
UKBB_ROOT = ROOT / "data/sourcedata"

# UKBB rfMRI parameters
UKBB_TR = 0.735  # seconds
UKBB_NUM_SUBJECTS = 740


def main():
    """Generate metadata for UKBB dataset."""
    # Load subject batch splits to determine train/val/test assignment
    splits_path = ROOT / "metadata/ukbb_subject_batch_splits.json"
    with splits_path.open("r") as f:
        batch_splits = json.load(f)

    # Create mapping: subject -> (split, batch_id)
    subject_to_split = {}
    for batch_id in range(20):
        batch_key = f"batch-{batch_id:02d}"
        subjects = batch_splits[batch_key]

        # Determine split based on batch_id
        if batch_id < 14:
            split = "train"
        elif batch_id < 17:
            split = "validation"
        else:
            split = "test"

        for sub in subjects:
            subject_to_split[sub] = (split, batch_id)

    # Find all meta.json files
    meta_paths = sorted(UKBB_ROOT.glob("sub-*_ses-*_mod-rfMRI.meta.json"))
    print(f"Found {len(meta_paths)} metadata files")
    assert len(meta_paths) == UKBB_NUM_SUBJECTS, f"Expected {UKBB_NUM_SUBJECTS} subjects"

    # Generate metadata records
    records = []
    for meta_path in meta_paths:
        # Parse filename: sub-{eid}_ses-{ses}_mod-{mod}.meta.json
        filename = meta_path.name
        parts = filename.replace(".meta.json", "").split("_")

        sub = parts[0].replace("sub-", "")
        ses = parts[1].replace("ses-", "")
        mod = parts[2].replace("mod-", "")

        # Load meta.json
        with meta_path.open("r") as f:
            meta = json.load(f)

        n_frames = meta["n_frames"]

        # Determine split and batch_id
        split, batch_id = subject_to_split[sub]

        # Path to bold.npy file (relative to sourcedata)
        bold_path = f"sub-{sub}_ses-{ses}_mod-{mod}.bold.npy"

        record = {
            "sub": sub,
            "mod": mod,
            "tr": UKBB_TR,
            "n_frames": n_frames,
            "path": bold_path,
            "split": split,
            "batch_id": batch_id,
        }
        records.append(record)

    # Create DataFrame
    df = pd.DataFrame(records)

    # Sort by subject ID
    df = df.sort_values("sub").reset_index(drop=True)

    # Print statistics
    print(f"\nTotal scans: {len(df)}")
    print("\nSplit counts:")
    print(df["split"].value_counts())
    print(f"\nBatch distribution:")
    print(df.groupby("split")["batch_id"].apply(lambda x: f"{x.min()}-{x.max()}"))

    # Save to parquet
    metadata_dir = ROOT / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    output_path = metadata_dir / "ukbb_metadata.parquet"
    df.to_parquet(output_path, index=False)

    print(f"\nSaved metadata to {output_path}")

    # Verify all paths exist
    missing_count = 0
    for path in df["path"]:
        full_path = UKBB_ROOT / path
        if not full_path.exists():
            print(f"Warning: {full_path} does not exist")
            missing_count += 1

    if missing_count == 0:
        print("All paths verified!")
    else:
        print(f"Warning: {missing_count} missing files")


if __name__ == "__main__":
    main()
