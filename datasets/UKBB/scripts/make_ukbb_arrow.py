import argparse
import logging
import sys
import tempfile
from pathlib import Path

import datasets as hfds
import numpy as np
import pandas as pd
from cloudpathlib import AnyPath, S3Path

import fmri_fm_eval.nisc as nisc
import fmri_fm_eval.readers as readers
import fmri_fm_eval.utils as ut

logging.basicConfig(
    format="[%(levelname)s %(asctime)s]: %(message)s",
    level=logging.INFO,
    datefmt="%y-%m-%d %H:%M:%S",
)

_logger = logging.getLogger(__name__)

ROOT = Path(__file__).parents[1]

# UKBB rfMRI parameters
UKBB_TR = 0.735
UKBB_NUM_FRAMES = 360  # Already correct length (~5 minutes)


def main(args):
    if args.space not in readers.DATA_DIMS:
        raise ValueError(f"Space {args.space} not supported. Available: {list(readers.DATA_DIMS.keys())}")

    # Use CIFTI files for all spaces
    source_root = ROOT / "data/sourcedata"
    reader = readers.READER_DICT[args.space]()

    out_root = AnyPath(args.out_root or (ROOT / "data/processed"))
    outdir = out_root / f"ukbb.{args.space}.arrow"

    _logger.info("Generating dataset: %s", outdir)
    if outdir.exists():
        _logger.info("Output %s exists; exiting.", outdir)
        return

    # Load metadata
    meta_df = pd.read_parquet(ROOT / "metadata/ukbb_metadata.parquet")
    target_df = pd.read_csv(ROOT / "metadata/ukbb_pheno_targets.csv", dtype={"Subject": str})
    target_df = target_df.set_index("Subject", drop=True)

    # Replace .bold.npy with .dtseries.nii for CIFTI paths
    meta_df["path"] = meta_df["path"].str.replace(".bold.npy", ".dtseries.nii")

    # Split paths by train/val/test
    path_splits = {
        "train": [],
        "validation": [],
        "test": [],
    }
    for split in ["train", "validation", "test"]:
        split_df = meta_df[meta_df["split"] == split]
        path_splits[split] = split_df["path"].tolist()
        _logger.info(f"Split ({split}): N={len(path_splits[split])} subjects")

    # Get data dimension for target space
    dim = readers.DATA_DIMS[args.space]

    # Define HuggingFace features schema
    features = hfds.Features(
        {
            "sub": hfds.Value("string"),
            "gender": hfds.Value("string"),  # "F" or "M"
            "age_q": hfds.Value("int32"),  # Age quantile bin (0, 1, 2)
            "path": hfds.Value("string"),
            "start": hfds.Value("int32"),
            "end": hfds.Value("int32"),
            "n_frames": hfds.Value("int32"),
            "tr": hfds.Value("float32"),
            "bold": hfds.Array2D(shape=(UKBB_NUM_FRAMES, dim), dtype="float16"),
            "mean": hfds.Array2D(shape=(1, dim), dtype="float32"),
            "std": hfds.Array2D(shape=(1, dim), dtype="float32"),
        }
    )

    # Generate the datasets with HuggingFace
    with tempfile.TemporaryDirectory(prefix="huggingface-") as tmpdir:
        dataset_dict = {}
        for split, paths in path_splits.items():
            dataset_dict[split] = hfds.Dataset.from_generator(
                generate_samples,
                features=features,
                gen_kwargs={
                    "paths": paths,
                    "root": source_root,
                    "target_df": target_df,
                    "reader": reader,
                },
                num_proc=args.num_proc,
                split=hfds.NamedSplit(split),
                cache_dir=tmpdir,
                fingerprint=f"ukbb-{args.space}-{split}",
            )
        dataset = hfds.DatasetDict(dataset_dict)

        _logger.info("Saving locally: %s", outdir)
        dataset.save_to_disk(outdir, max_shard_size="300MB")


def generate_samples(
    paths: list[str],
    *,
    root: Path,
    target_df: pd.DataFrame,
    reader: readers.Reader,
):
    """Generate samples from UKBB files."""
    for path in paths:
        fullpath = root / path
        filename = Path(path).name
        sub = filename.split("_")[0].replace("sub-", "")

        bold = reader(fullpath)  # (T, dim) - reader handles space transformation

        if bold.shape[0] > UKBB_NUM_FRAMES:
            bold = bold[:UKBB_NUM_FRAMES]

        assert bold.shape[0] == UKBB_NUM_FRAMES, f"Expected {UKBB_NUM_FRAMES} TRs, got {bold.shape[0]}"
        series, mean, std = nisc.scale(bold.astype(np.float32))

        row = target_df.loc[sub]
        gender = row["Gender"]  # Already mapped to "F"/"M" in make_ukbb_targets.py
        age_q = int(row["Age_Q"])

        sample = {
            "sub": sub,
            "gender": gender,
            "age_q": age_q,
            "path": str(path),
            "start": 0,
            "end": UKBB_NUM_FRAMES,
            "n_frames": UKBB_NUM_FRAMES,
            "tr": UKBB_TR,
            "bold": series.astype(np.float16),
            "mean": mean.astype(np.float32),
            "std": std.astype(np.float32),
        }
        yield sample


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=str, default=None)
    parser.add_argument(
        "--space", type=str, default="flat", choices=list(readers.READER_DICT.keys())
    )
    parser.add_argument("--num_proc", "-j", type=int, default=8)
    args = parser.parse_args()
    sys.exit(main(args))
