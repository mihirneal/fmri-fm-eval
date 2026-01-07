import os
from pathlib import Path

import datasets as hfds

from fmri_fm_eval.datasets.base import HFDataset
from fmri_fm_eval.datasets.registry import register_dataset

AABC_ROOT = str(Path(__file__).parents[3] / "datasets" / "AABC")

AABC_TARGET_MAP_DICT = {
    # Demographics
    "sex": "aabc_target_map_sex.json",
    "age": "aabc_target_map_age_open.json",
    # NEO-FFI Personality
    "neo_n": "aabc_target_map_neo_n.json",
    "neo_e": "aabc_target_map_neo_e.json",
    "neo_o": "aabc_target_map_neo_o.json",
    "neo_a": "aabc_target_map_neo_a.json",
    "neo_c": "aabc_target_map_neo_c.json",
    # Cognitive Composites
    "fluid_iq": "aabc_target_map_FluidIQ_Tr35_60y.json",
    "cryst_iq": "aabc_target_map_CrystIQ_Tr35_60y.json",
    "memory": "aabc_target_map_Memory_Tr35_60y.json",
    # RAVLT Memory
    "ravlt_immediate": "aabc_target_map_ravlt_immediate_recall.json",
    "ravlt_delay": "aabc_target_map_ravlt_delay_completion.json",
    "ravlt_learning": "aabc_target_map_ravlt_learning_score.json",
    # Executive Function
    "dccs": "aabc_target_map_tlbx_dccs_uncorrected_standard_score.json",
    "flanker": "aabc_target_map_tlbx_fica_uncorrected_standard_score.json",
    # Motor Function
    "grip": "aabc_target_map_tlbx_grip_uncorrected_standard_scores_dominant.json",
    "walk": "aabc_target_map_tlbx_walk_2_uncorrected_standard_score.json",
}


def _create_aabc(space: str, target: str, **kwargs):
    target_key = "sub"
    target_map_path = AABC_TARGET_MAP_DICT[target]
    target_map_path = f"{AABC_ROOT}/metadata/targets/{target_map_path}"

    dataset_dict = {}
    splits = ["train", "validation", "test"]
    for split in splits:
        url = f"{AABC_ROOT}/aabc.{space}.arrow/{split}"
        dataset = hfds.load_dataset("arrow", data_files=f"{url}/*.arrow", split="train", **kwargs)
        dataset = HFDataset(dataset, target_key=target_key, target_map_path=target_map_path)
        dataset_dict[split] = dataset

    return dataset_dict


# Demographics
@register_dataset
def aabc_sex(space: str, **kwargs):
    return _create_aabc(space, target="sex", **kwargs)


@register_dataset
def aabc_age(space: str, **kwargs):
    return _create_aabc(space, target="age", **kwargs)


# NEO-FFI Personality
@register_dataset
def aabc_neo_n(space: str, **kwargs):
    return _create_aabc(space, target="neo_n", **kwargs)


@register_dataset
def aabc_neo_e(space: str, **kwargs):
    return _create_aabc(space, target="neo_e", **kwargs)


@register_dataset
def aabc_neo_o(space: str, **kwargs):
    return _create_aabc(space, target="neo_o", **kwargs)


@register_dataset
def aabc_neo_a(space: str, **kwargs):
    return _create_aabc(space, target="neo_a", **kwargs)


@register_dataset
def aabc_neo_c(space: str, **kwargs):
    return _create_aabc(space, target="neo_c", **kwargs)


# Cognitive Composites
@register_dataset
def aabc_fluid_iq(space: str, **kwargs):
    return _create_aabc(space, target="fluid_iq", **kwargs)


@register_dataset
def aabc_cryst_iq(space: str, **kwargs):
    return _create_aabc(space, target="cryst_iq", **kwargs)


@register_dataset
def aabc_memory(space: str, **kwargs):
    return _create_aabc(space, target="memory", **kwargs)


# RAVLT Memory
@register_dataset
def aabc_ravlt_immediate(space: str, **kwargs):
    return _create_aabc(space, target="ravlt_immediate", **kwargs)


@register_dataset
def aabc_ravlt_delay(space: str, **kwargs):
    return _create_aabc(space, target="ravlt_delay", **kwargs)


@register_dataset
def aabc_ravlt_learning(space: str, **kwargs):
    return _create_aabc(space, target="ravlt_learning", **kwargs)


# Executive Function
@register_dataset
def aabc_dccs(space: str, **kwargs):
    return _create_aabc(space, target="dccs", **kwargs)


@register_dataset
def aabc_flanker(space: str, **kwargs):
    return _create_aabc(space, target="flanker", **kwargs)


# Motor Function
@register_dataset
def aabc_grip(space: str, **kwargs):
    return _create_aabc(space, target="grip", **kwargs)


@register_dataset
def aabc_walk(space: str, **kwargs):
    return _create_aabc(space, target="walk", **kwargs)
