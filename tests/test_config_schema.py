from __future__ import annotations

from omegaconf import OmegaConf
from siamram.config import flatten_subsystem_overrides


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
