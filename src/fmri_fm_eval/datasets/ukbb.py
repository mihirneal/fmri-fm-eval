import os

from fmri_fm_eval.datasets.base import HFDataset, load_arrow_dataset
from fmri_fm_eval.datasets.registry import register_dataset

UKBB_ROOT = os.getenv("UKBB_ROOT", "s3://ukbb/eval")


def _create_ukbb(
    space: str,
    target_key: str | None = None,
    **kwargs,
):
    """Create UKBB dataset with specified target.

    Args:
        space: Anatomical space (e.g., "flat")
        target_key: Target variable name (e.g., "age_q", "gender")
        **kwargs: Additional arguments for dataset loading

    Returns:
        Dictionary of train/validation/test datasets
    """
    dataset_dict = {}
    splits = ["train", "validation", "test"]
    for split in splits:
        url = f"{UKBB_ROOT}/ukbb.{space}.arrow/{split}"
        dataset = load_arrow_dataset(url)
        dataset = HFDataset(dataset, target_key=target_key)
        dataset_dict[split] = dataset

    return dataset_dict


@register_dataset
def ukbb_age(space: str, **kwargs):
    """UKBB age prediction (3-class classification).

    Age bins (quantile-based):
        - Bin 0: 40-51 years (~267 subjects)
        - Bin 1: 52-59 years (~243 subjects)
        - Bin 2: 60-70 years (~230 subjects)

    Total: 740 subjects
    Split: 518 train / 111 validation / 111 test
    """
    return _create_ukbb(space, target_key="age_q", **kwargs)


@register_dataset
def ukbb_gender(space: str, **kwargs):
    """UKBB gender prediction (binary classification).

    Gender distribution:
        - F (Female): 399 subjects (54%)
        - M (Male): 341 subjects (46%)

    Total: 740 subjects
    Split: 518 train / 111 validation / 111 test
    """
    return _create_ukbb(space, target_key="gender", **kwargs)
