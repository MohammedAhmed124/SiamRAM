from __future__ import annotations

from omegaconf import OmegaConf
from models.siamram.config import (
    flatten_ram_tracker_config,
    flatten_subsystem_overrides,
)


def test_flatten_subsystem_overrides_descriptor_and_reacq() -> None:
    cfg = OmegaConf.create(
        {
            "ram_tracker_subsystems": {
                "descriptor": {
                    "osnet_pretrained_checkpoint": "reid_market1501",
                    "osnet_model_name": "osnet_x1_0",
                },
                "reacquisition": {
                    "threshold": 0.72,
                    "confirm_frames": 4,
                    "occ_siam_threshold": 0.81,
                },
            }
        }
    )
    ram, exp = flatten_subsystem_overrides(cfg)
    assert ram["osnet_pretrained_checkpoint"] == "reid_market1501"
    assert ram["reacq_threshold"] == 0.72
    assert ram["reacq_confirm_frames"] == 4
    assert ram["occ_siam_reacq_threshold"] == 0.81
    assert exp == {}


def test_flatten_subsystem_overrides_invalid_checkpoint() -> None:
    cfg = OmegaConf.create(
        {
            "ram_tracker_subsystems": {
                "descriptor": {
                    "osnet_pretrained_checkpoint": "bad_checkpoint_name",
                }
            }
        }
    )
    try:
        flatten_subsystem_overrides(cfg)
    except ValueError:
        return
    raise AssertionError("Expected ValueError for invalid checkpoint choice")


def test_flatten_ram_tracker_config_from_nested_groups() -> None:
    cfg = OmegaConf.create(
        {
            "ram_tracker": {
                "descriptor": {
                    "descriptor_backend": "siamese",
                    "siamese_feature_source": "neck",
                    "siamese_comparison_mode": "xcorr",
                },
                "occlusion": {
                    "reacquisition": {
                        "reacq_threshold": 0.72,
                        "reacq_confirm_frames": 4,
                    }
                },
            }
        }
    )
    flat = flatten_ram_tracker_config(cfg)
    assert flat["descriptor_backend"] == "siamese"
    assert flat["siamese_feature_source"] == "neck"
    assert flat["siamese_comparison_mode"] == "xcorr"
    assert flat["reacq_threshold"] == 0.72
    assert flat["reacq_confirm_frames"] == 4


def test_flatten_ram_tracker_config_duplicate_leaf_key_raises() -> None:
    cfg = OmegaConf.create(
        {
            "ram_tracker": {
                "descriptor": {"descriptor_backend": "osnet"},
                "occlusion": {"descriptor_backend": "siamese"},
            }
        }
    )
    try:
        flatten_ram_tracker_config(cfg)
    except ValueError:
        return
    raise AssertionError("Expected ValueError for duplicate ram_tracker leaf key")
