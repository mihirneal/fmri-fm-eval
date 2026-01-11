"""Create ADNI evaluation dataset with TR resampling (Arrow format).

This script creates HuggingFace Arrow datasets for ADNI resting-state fMRI:
- Resamples all BOLD data from original TR to 0.72s
- Takes single 500 TR sample per session (no windowing)
- Uses existing Train/Val/Test split from metadata CSV

Supports parcellations: schaefer400, schaefer400_tians3, flat, a424, fslr91k
"""
import argparse
import json
import logging
import sys
import tempfile
from pathlib import Path

import datasets as hfds
import numpy as np
import pandas as pd

import scipy.interpolate
import scipy.signal

import fmri_fm_eval.nisc as nisc
import fmri_fm_eval.readers as readers

logging.basicConfig(
    format="[%(levelname)s %(asctime)s]: %(message)s",
    level=logging.INFO,
    datefmt="%y-%m-%d %H:%M:%S",
)
logging.getLogger("nibabel").setLevel(logging.ERROR)

_logger = logging.getLogger(__name__)

ROOT = Path(__file__).parents[1]
ADNI_FMRIPREP = Path("/teamspace/studios/this_studio/ADNI_fmriprep")
ADNI_OUTPUT = ADNI_FMRIPREP / "output"
ADNI_CSV = ADNI_FMRIPREP / "valid_adni_benchmark_split_final.csv"
ADNI_TR_JSON = ADNI_FMRIPREP / "valid_subjects_sessions_with_metadata.json"

# Target TR for resampling (matches HCP/AABC)
TARGET_TR = 0.72

# Maximum TRs to keep after resampling (360 seconds at 0.72s TR)
MAX_TRS = 500

# Split mapping from CSV
SPLIT_MAP = {
    "Train": "train",
    "Val": "validation",
    "Test": "test",
}


def resample_timeseries_safe(
    series: np.ndarray,
    tr: float,
    new_tr: float,
    kind: str = "cubic",
    antialias: bool = True,
) -> np.ndarray:
    """Resample a time series to a target TR with safe bounds handling.

    This is a wrapper around nisc.resample_timeseries that fixes floating point
    precision issues at the interpolation boundaries.
    """
    if tr == new_tr:
        return series

    fs = 1.0 / tr
    new_fs = 1.0 / new_tr

    # Anti-aliasing low-pass filter (from nisc.resample_timeseries)
    if antialias and new_fs < fs:
        q = fs / new_fs
        sos = scipy.signal.cheby1(8, 0.05, 0.8 / q, output="sos")
        series = scipy.signal.sosfiltfilt(sos, series, axis=0, padtype="even")

    # Original time points
    n_orig = len(series)
    x = tr * np.arange(n_orig)
    max_t = x[-1]  # Maximum time in original data

    # Compute new time points, ensuring we don't exceed original range
    # Use floor division to be safe with floating point
    new_length = int(np.floor(max_t / new_tr)) + 1
    new_x = new_tr * np.arange(new_length)

    # Clip to original range (handles floating point precision issues)
    new_x = np.clip(new_x, 0, max_t)

    if kind == "pchip":
        interp = scipy.interpolate.PchipInterpolator(x, series, axis=0)
    else:
        interp = scipy.interpolate.interp1d(x, series, kind=kind, axis=0)

    series = interp(new_x)
    return series


def ptid_to_sub(ptid: str) -> str:
    """Convert PTID format to sub directory format.

    Example: "168_S_6049" -> "168S6049"
    """
    return ptid.replace("_", "")


def scandate_to_ses(scandate: str) -> str:
    """Convert scandate format to session directory format.

    Example: "2020-10-12" -> "20201012"
    """
    return scandate.replace("-", "")


def main(args):
    outdir = ROOT / f"data/processed/adni.{args.space}.arrow"
    _logger.info("Generating dataset: %s", outdir)
    if outdir.exists() and not args.overwrite:
        _logger.warning("Output %s exists; exiting. Use --overwrite to replace.", outdir)
        return 1

    # Check for a424 CIFTI availability
    if args.space == "a424":
        try:
            nisc.fetch_a424(cifti=True)
        except Exception as exc:
            _logger.error(
                "A424 from CIFTI requires a CIFTI parcellation. "
                "Set A424_CIFTI_PATH or provide volumetric NIfTI inputs. (%s)",
                exc,
            )
            return 1

    # Load metadata CSV
    df = pd.read_csv(ADNI_CSV)
    _logger.info("Loaded %d samples from CSV", len(df))

    # Load TR metadata
    with ADNI_TR_JSON.open() as f:
        tr_metadata = json.load(f)
    _logger.info("Loaded TR metadata for %d subjects", tr_metadata["total_subjects"])

    # Build TR lookup: {(sub, ses): TR}
    tr_lookup = {}
    for sub, sessions in tr_metadata["subjects"].items():
        for sess in sessions:
            ses = sess["session_id"]
            tr_lookup[(sub, ses)] = sess["TR"]

    # Build samples by split
    samples_by_split = {"train": [], "validation": [], "test": []}
    missing_tr = 0
    missing_file = 0

    for _, row in df.iterrows():
        ptid = row["PTID"]
        scandate = row["SCANDATE_Imaging"]
        split = SPLIT_MAP[row["Split"]]

        sub = ptid_to_sub(ptid)
        ses = scandate_to_ses(scandate)

        # Build CIFTI path
        cifti_path = (
            ADNI_OUTPUT / f"sub-{sub}" / f"ses-{ses}" / "func" /
            f"sub-{sub}_ses-{ses}_task-rest_space-fsLR_den-91k_bold.dtseries.nii"
        )

        # Check if file exists
        if not cifti_path.exists():
            _logger.debug("Missing CIFTI: %s", cifti_path)
            missing_file += 1
            continue

        # Get TR for this session
        # TR lookup key uses format from JSON: sub without prefix, ses without prefix
        tr_key = (sub, ses)
        if tr_key not in tr_lookup:
            _logger.debug("Missing TR for %s/%s", sub, ses)
            missing_tr += 1
            continue

        original_tr = tr_lookup[tr_key]

        sample = {
            "sub": sub,
            "visit": ses,
            "mod": "MR",
            "task": "rest",
            "path": str(cifti_path.relative_to(ADNI_OUTPUT)),
            "fullpath": str(cifti_path),
            "original_tr": original_tr,
            "ptid": ptid,
            "scandate": scandate,
        }
        samples_by_split[split].append(sample)

    if missing_file > 0:
        _logger.warning("Missing %d CIFTI files", missing_file)
    if missing_tr > 0:
        _logger.warning("Missing TR for %d samples", missing_tr)

    for split, samples in samples_by_split.items():
        _logger.info("Num samples (%s): %d", split, len(samples))

    # Load reader for target space
    if args.space == "a424":
        reader = readers.a424_reader(cifti=True)
    else:
        reader = readers.READER_DICT[args.space]()
    dim = readers.DATA_DIMS[args.space]
    _logger.info("Using reader for space '%s' with dimension: %d", args.space, dim)

    # Features
    features = hfds.Features(
        {
            "sub": hfds.Value("string"),
            "visit": hfds.Value("string"),
            "mod": hfds.Value("string"),
            "task": hfds.Value("string"),
            "path": hfds.Value("string"),
            "start": hfds.Value("int32"),
            "end": hfds.Value("int32"),
            "tr": hfds.Value("float32"),
            "segment": hfds.Value("int32"),
            "bold": hfds.Array2D(shape=(None, dim), dtype="float16"),
            "mean": hfds.Array2D(shape=(1, dim), dtype="float32"),
            "std": hfds.Array2D(shape=(1, dim), dtype="float32"),
        }
    )

    writer_batch_size = args.writer_batch_size
    if writer_batch_size is None and args.space == "flat":
        writer_batch_size = 16

    # Generate datasets
    with tempfile.TemporaryDirectory(prefix="huggingface-") as tmpdir:
        dataset_dict = {}
        for split, samples in samples_by_split.items():
            if not samples:
                _logger.warning("No samples for split %s; skipping.", split)
                continue
            dataset_dict[split] = hfds.Dataset.from_generator(
                generate_samples,
                features=features,
                gen_kwargs={"samples": samples, "reader": reader, "dim": dim},
                num_proc=args.num_proc,
                split=hfds.NamedSplit(split),
                cache_dir=tmpdir,
                writer_batch_size=writer_batch_size,
            )
        dataset = hfds.DatasetDict(dataset_dict)

        outdir.parent.mkdir(exist_ok=True, parents=True)
        dataset.save_to_disk(outdir, max_shard_size="300MB")

    _logger.info("Dataset saved to: %s", outdir)
    return 0


def generate_samples(samples: list[dict], *, reader, dim: int):
    """Generate samples for evaluation dataset with TR resampling."""
    for sample_info in samples:
        fullpath = sample_info["fullpath"]
        original_tr = sample_info["original_tr"]

        try:
            # Read CIFTI data
            series = reader(fullpath)
            T_orig, D = series.shape
            assert D == dim, f"Expected dim {dim}, got {D} for {fullpath}"

            # Resample to target TR using safe function
            if original_tr != TARGET_TR:
                series = resample_timeseries_safe(
                    series, tr=original_tr, new_tr=TARGET_TR
                )

            T_resampled = len(series)

            # Take first MAX_TRS (or all if shorter)
            n_trs = min(MAX_TRS, T_resampled)
            series = series[:n_trs]

            # Z-score normalization
            series, mean, std = nisc.scale(series)

            yield {
                "sub": sample_info["sub"],
                "visit": sample_info["visit"],
                "mod": sample_info["mod"],
                "task": sample_info["task"],
                "path": sample_info["path"],
                "start": 0,
                "end": n_trs,
                "tr": TARGET_TR,
                "segment": 0,  # Single sample per session, no windowing
                "bold": series.astype(np.float16),
                "mean": mean.astype(np.float32),
                "std": std.astype(np.float32),
            }

        except Exception as exc:
            _logger.warning("Failed to process %s: %s", fullpath, exc)
            continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create ADNI evaluation dataset")
    parser.add_argument(
        "--space",
        type=str,
        default="schaefer400",
        choices=list(readers.READER_DICT),
        help="Target anatomical space for processing (default: schaefer400)"
    )
    parser.add_argument(
        "--num_proc",
        "-j",
        type=int,
        default=32,
        help="Number of parallel processes"
    )
    parser.add_argument(
        "--writer_batch_size",
        type=int,
        default=None,
        help="Arrow writer batch size (default: 16 for flat, otherwise datasets default)"
    )
    parser.add_argument(
        "--overwrite",
        "-x",
        action="store_true",
        help="Overwrite existing output directory"
    )
    args = parser.parse_args()
    sys.exit(main(args))
