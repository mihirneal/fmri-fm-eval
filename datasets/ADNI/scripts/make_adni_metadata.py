import logging
from pathlib import Path

import datasets as hfds
import nibabel as nib

logging.basicConfig(
    format="[%(levelname)s %(asctime)s]: %(message)s",
    level=logging.INFO,
    datefmt="%y-%m-%d %H:%M:%S",
)
logging.getLogger("nibabel").setLevel(logging.ERROR)

_logger = logging.getLogger(__name__)

NUM_PROC = 16

ROOT = Path(__file__).parents[1]
ADNI_ROOT = Path("/teamspace/studios/this_studio/ADNI_fmriprep/output")


def main():
    """Generate metadata for ADNI dataset."""
    _logger.info("Finding CIFTI files in %s", ADNI_ROOT)

    # Find all CIFTI files in ADNI dataset
    series_paths = sorted(
        ADNI_ROOT.rglob("*_space-fsLR_den-91k_bold.dtseries.nii")
    )

    _logger.info("Found %d scan files", len(series_paths))

    features = hfds.Features(
        {
            "sub": hfds.Value("string"),
            "visit": hfds.Value("string"),
            "mod": hfds.Value("string"),
            "task": hfds.Value("string"),
            "tr": hfds.Value("float32"),
            "n_frames": hfds.Value("int32"),
            "path": hfds.Value("string"),
        }
    )

    dataset = hfds.Dataset.from_generator(
        generate_metadata,
        features=features,
        gen_kwargs={"paths": series_paths},
        num_proc=NUM_PROC,
    )

    # Create metadata directory if it doesn't exist
    metadata_dir = ROOT / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    output_path = metadata_dir / "adni_metadata.parquet"
    dataset.to_parquet(output_path)

    _logger.info("Saved metadata for %d scans to %s", len(dataset), output_path)

    # Print statistics
    _logger.info("\n=== Metadata Statistics ===")

    # Task counts (should all be "rest")
    task_counts = {}
    tr_values = []
    subjects = set()

    for sample in dataset:
        task = sample["task"]
        task_counts[task] = task_counts.get(task, 0) + 1
        tr_values.append(sample["tr"])
        subjects.add(sample["sub"])

    _logger.info("Unique subjects: %d", len(subjects))
    _logger.info("Total scans: %d", len(dataset))

    _logger.info("\nScan counts by task:")
    for task, count in sorted(task_counts.items()):
        _logger.info("  %s: %d", task, count)

    # TR distribution
    tr_unique = sorted(set(tr_values))
    _logger.info("\nTR distribution (%d unique values):", len(tr_unique))
    for tr in tr_unique:
        count = tr_values.count(tr)
        _logger.info("  %.5f s: %d scans", tr, count)

    # Check for very short scans
    short_scans = [s for s in dataset if s["n_frames"] < 100]
    if short_scans:
        _logger.warning("\nFound %d scans with < 100 frames:", len(short_scans))
        for s in short_scans[:10]:  # Show first 10
            _logger.warning("  %s: %d frames (TR=%.2fs)", s["path"], s["n_frames"], s["tr"])


def generate_metadata(paths: list[str]):
    """Generate metadata for each CIFTI file."""
    for path in paths:
        path = Path(path)
        meta = parse_adni_metadata(path)

        # Load CIFTI header to get number of frames and TR
        img = nib.load(path)
        n_frames = img.shape[0]

        # Extract TR from CIFTI header
        tr = img.header.matrix.get_index_map(0).series_step

        sample = {
            **meta,
            "tr": float(tr),
            "n_frames": int(n_frames),
            "path": str(path.relative_to(ADNI_ROOT)),
        }
        yield sample


def parse_adni_metadata(path: Path) -> dict[str, str]:
    """Parse ADNI metadata from file path.

    Path structure:
        sub-168S6049/ses-20201012/func/sub-168S6049_ses-20201012_task-rest_space-fsLR_den-91k_bold.dtseries.nii

    Returns:
        dict with keys:
        - sub: Subject ID (e.g., "168S6049")
        - visit: Session ID (e.g., "20201012")
        - mod: Modality (always "MR")
        - task: Task type (always "rest")
    """
    # Path structure: sub-{id}/ses-{date}/func/{filename}
    parts = path.parts

    # Extract subject ID: sub-168S6049 → 168S6049
    subject_dir = parts[-4]
    sub = subject_dir.replace("sub-", "")

    # Extract session ID: ses-20201012 → 20201012
    session_dir = parts[-3]
    visit = session_dir.replace("ses-", "")

    # ADNI only has resting-state fMRI
    mod = "MR"
    task = "rest"

    metadata = {
        "sub": sub,
        "visit": visit,
        "mod": mod,
        "task": task,
    }
    return metadata


if __name__ == "__main__":
    main()
