import json
import logging
import os
from pathlib import Path

import pandas as pd

logging.basicConfig(
    format="[%(levelname)s %(asctime)s]: %(message)s",
    level=logging.INFO,
    datefmt="%y-%m-%d %H:%M:%S",
)

_logger = logging.getLogger(__name__)

ROOT = Path(__file__).parents[1]
AABC_ROOT = Path(os.getenv("AABC_ROOT", "/teamspace/studios/this_studio/AABC_data"))
AABC_CSV_PATH = AABC_ROOT / "AABC_subjects_2026_01_03_14_21_56.csv"

# Gender mapping (classification target)
GENDER_MAP = {
    "F": 0,
    "M": 1,
}
GENDER_CLASSES = ["F", "M"]

# Phenotypic/Cognitive targets
#
# Demographics:
# - sex: Binary gender (M/F)
# - age_open: Age in years
#
# Cognitive Composites:
# - Memory_Tr35_60y: Memory composite score
# - FluidIQ_Tr35_60y: Fluid intelligence composite
# - CrystIQ_Tr35_60y: Crystallized intelligence composite
#
# Task-Specific Predictions:
# CARIT (Executive Function):
# - tlbx_dccs_uncorrected_standard_score: Dimensional Change Card Sort
# - tlbx_fica_uncorrected_standard_score: Flanker Inhibitory Control
#
# FACENAME (Memory):
# - ravlt_immediate_recall: RAVLT immediate recall
# - ravlt_delay_completion: RAVLT delayed recall
# - ravlt_learning_score: RAVLT learning slope
#
# VISMOTOR (Motor):
# - tlbx_grip_uncorrected_standard_scores_dominant: Grip strength (dominant hand)
# - tlbx_walk_2_uncorrected_standard_score: Walking speed
#
# Personality (NEO-FFI Big Five):
# - neo_n: Neuroticism
# - neo_e: Extraversion
# - neo_o: Openness
# - neo_a: Agreeableness
# - neo_c: Conscientiousness

TARGETS = [
    # Demographics
    "sex",
    "age_open",
    # Cognitive composites
    "Memory_Tr35_60y",
    "FluidIQ_Tr35_60y",
    "CrystIQ_Tr35_60y",
    # CARIT task (Executive Function)
    "tlbx_dccs_uncorrected_standard_score",
    "tlbx_fica_uncorrected_standard_score",
    # FACENAME task (Memory)
    "ravlt_immediate_recall",
    "ravlt_delay_completion",
    "ravlt_learning_score",
    # VISMOTOR task (Motor)
    "tlbx_grip_uncorrected_standard_scores_dominant",
    "tlbx_walk_2_uncorrected_standard_score",
    # NEO-FFI Personality
    "neo_n",
    "neo_e",
    "neo_o",
    "neo_a",
    "neo_c",
]


def main():
    # Load AABC phenotypic data
    df = pd.read_csv(AABC_CSV_PATH, usecols=["id_event"] + TARGETS)

    # Convert all columns except 'sex' to numeric, coercing errors to NaN
    for col in TARGETS:
        if col != "sex":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Extract subject ID from id_event (HCA6000030_V1 -> HCA6000030)
    df["sub"] = df["id_event"].str.rsplit("_", n=1).str[0]

    # For subjects with multiple visits, use data from first visit
    # (longitudinal changes are minimal for most phenotypes)
    df = df.drop_duplicates(subset="sub", keep="first")
    df = df.set_index("sub")

    outdir = ROOT / "metadata/target"
    outdir.mkdir(exist_ok=True, parents=True)

    for target in TARGETS:
        outpath = outdir / f"aabc_target_map_{target}.json"
        infopath = outdir / f"aabc_target_info_{target}.json"
        series = df[target]
        na_mask = series.isna()
        series = series.loc[~na_mask]
        na_count = int(na_mask.sum())

        if target == "sex":
            targets = series.map(GENDER_MAP)
            counts = targets.value_counts().sort_index().tolist()
            info = {
                "target": target,
                "na_count": na_count,
                "classes": GENDER_CLASSES,
                "label_counts": counts,
            }
        else:
            targets = series.astype(float)
            info = {
                "target": target,
                "na_count": na_count,
                "min": float(targets.min()),
                "max": float(targets.max()),
                "mean": float(targets.mean()),
                "median": float(targets.median()),
                "std": float(targets.std()),
            }

        targets = targets.to_dict()
        _logger.info(json.dumps(info))

        with outpath.open("w") as f:
            print(json.dumps(targets, indent=4), file=f)

        with infopath.open("w") as f:
            print(json.dumps(info, indent=4), file=f)

    _logger.info(f"Created {len(TARGETS)} target files in {outdir}")


if __name__ == "__main__":
    main()
