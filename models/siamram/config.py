"""
Typed config helpers for SiamRAM tracker options.

This module supports three config layouts:
1) Modern nested groups under `ram_tracker` (preferred).
2) Legacy flat `ram_tracker` keys.
3) Compatibility blocks (`ram_tracker_subsystems`, `ram_tracker_experiment`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

from omegaconf import DictConfig, OmegaConf

OSNET_CHECKPOINT_CHOICES = {
    "imagenet",
    "reid_market1501",
    "reid_dukemtmcreid",
    "reid_msmt17",
    "reid_msmt17_combineall",
    "custom",
}


@dataclass
class DescriptorConfig:
    backend: str = "osnet"
    osnet_model_name: str = "osnet_x1_0"
    osnet_pretrained_checkpoint: str = "imagenet"
    osnet_model_path: str = ""
    osnet_device: str = "auto"
    osnet_max_candidate_batch: int = 0
    # Supported backends: "osnet", "siamese", "pixel descriptors".
    # Only consulted when backend == "siamese". Picks which SiamABC layer to
    # pool into the appearance embedding. "neck" → ~256-d, "encoder" → ~1024-d.
    siamese_feature_source: str = "neck"
    # Only consulted when backend == "siamese". Selects how two descriptors
    # are compared:
    #   "xcorr"  — keep (C, H, W) feature map, compare via 2D cross-correlation.
    #              Matches how SiamABC was trained. Default.
    #   "pooled" — global avg pool to a (C,) vector, compare via cosine.
    siamese_comparison_mode: str = "xcorr"


@dataclass
class ReacquisitionConfig:
    threshold: float = 0.55
    confirm_frames: int = 1
    occ_siam_threshold: float = 0.80


@dataclass
class GmcPriorConfig:
    enabled: bool = True
    require_reliable_h: bool = True
    max_translation_frac: float = 0.25
    min_scale: float = 0.70
    max_scale: float = 1.40
    max_rotation_deg: float = 25.0
    max_corner_displacement_frac: float = 0.25


@dataclass
class TrackerSubsystemConfig:
    descriptor: Optional[DescriptorConfig] = None
    reacquisition: Optional[ReacquisitionConfig] = None
    gmc_prior: Optional[GmcPriorConfig] = None


def _to_mapping(value: Any) -> Mapping[str, Any]:
    if value is None:
        return {}
    if isinstance(value, DictConfig):
        data = OmegaConf.to_container(value, resolve=True)
    else:
        data = value
    if not isinstance(data, Mapping):
        return {}
    return data


def _validate_subsystems(raw_subsystems: Mapping[str, Any]) -> TrackerSubsystemConfig:
    merged = OmegaConf.merge(
        OmegaConf.structured(TrackerSubsystemConfig),
        OmegaConf.create(dict(raw_subsystems)),
    )
    obj = OmegaConf.to_object(merged)
    assert isinstance(obj, TrackerSubsystemConfig)
    return obj


def _validate_osnet_checkpoint(name: str) -> None:
    if name not in OSNET_CHECKPOINT_CHOICES:
        options = ", ".join(sorted(OSNET_CHECKPOINT_CHOICES))
        raise ValueError(
            "Unsupported osnet_pretrained_checkpoint="
            f"'{name}'. Expected one of: {options}."
        )


def _flatten_nested_mapping(
    raw: Mapping[str, Any],
    *,
    scope: str,
    conflict_mode: str = "error",
) -> dict[str, Any]:
    """
    Flatten nested mapping leaves to a single dict keyed by leaf names.

    Example:
        {"descriptor": {"backend": "siamese"}} -> {"backend": "siamese"}

    Notes:
        - Keys beginning with "_" are ignored (reserved metadata/comments).
        - If two leaves produce the same key, conflict behavior is controlled by
          `conflict_mode`:
            * "error" -> raise ValueError with both source paths.
            * "last"  -> keep the later value.
    """
    out: dict[str, Any] = {}
    origins: dict[str, tuple[str, ...]] = {}

    def _walk(node: Mapping[str, Any], path: tuple[str, ...]) -> None:
        for key, value in node.items():
            if str(key).startswith("_"):
                continue

            key_str = str(key)
            child_path = (*path, key_str)
            if isinstance(value, (Mapping, DictConfig)):
                child_map = _to_mapping(value)
                _walk(child_map, child_path)
                continue

            if key_str in out and conflict_mode == "error":
                prev = ".".join(origins[key_str])
                curr = ".".join(child_path)
                raise ValueError(
                    f"Duplicate ram_tracker leaf key '{key_str}' while flattening "
                    f"{scope}: '{prev}' and '{curr}'. Keep only one."
                )

            out[key_str] = value
            origins[key_str] = child_path

    _walk(raw, ())
    return out


def flatten_ram_tracker_config(config: Any) -> dict[str, Any]:
    """
    Produce one flat kwargs dict for SiamRAMExperimentTracker.

    Merge order (last writer wins for compatibility blocks):
        1) `ram_tracker` (nested or flat; duplicates inside this block error)
        2) legacy `ram_tracker_subsystems`
        3) legacy `ram_tracker_experiment`
    """
    ram_tracker_raw = _to_mapping(getattr(config, "ram_tracker", None))
    ram_kwargs = _flatten_nested_mapping(
        ram_tracker_raw,
        scope="ram_tracker",
        conflict_mode="error",
    )

    legacy_ram, legacy_exp = flatten_subsystem_overrides(config)
    ram_kwargs.update(legacy_ram)
    ram_kwargs.update(legacy_exp)

    legacy_exp_raw = _to_mapping(getattr(config, "ram_tracker_experiment", None))
    if legacy_exp_raw:
        legacy_flat = _flatten_nested_mapping(
            legacy_exp_raw,
            scope="ram_tracker_experiment",
            conflict_mode="last",
        )
        ram_kwargs.update(legacy_flat)

    osnet_ckpt = str(ram_kwargs.get("osnet_pretrained_checkpoint", "")).strip()
    if osnet_ckpt:
        _validate_osnet_checkpoint(osnet_ckpt)

    return ram_kwargs


def flatten_subsystem_overrides(config: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Build legacy flat kwargs from nested `ram_tracker_subsystems`.

    Returns:
        (ram_tracker_overrides, ram_tracker_experiment_overrides)
    """
    raw_subsystems = _to_mapping(getattr(config, "ram_tracker_subsystems", None))
    if not raw_subsystems:
        return {}, {}

    subsystems = _validate_subsystems(raw_subsystems)
    ram_overrides: dict[str, Any] = {}
    exp_overrides: dict[str, Any] = {}

    if subsystems.descriptor is not None:
        desc = subsystems.descriptor
        _validate_osnet_checkpoint(desc.osnet_pretrained_checkpoint)
        ram_overrides.update(
            {
                "descriptor_backend": desc.backend,
                "osnet_model_name": desc.osnet_model_name,
                "osnet_pretrained_checkpoint": desc.osnet_pretrained_checkpoint,
                "osnet_model_path": desc.osnet_model_path,
                "osnet_device": desc.osnet_device,
                "osnet_max_candidate_batch": int(desc.osnet_max_candidate_batch),
                "siamese_feature_source": desc.siamese_feature_source,
                "siamese_comparison_mode": desc.siamese_comparison_mode,
            }
        )

    if subsystems.reacquisition is not None:
        reacq = subsystems.reacquisition
        ram_overrides.update(
            {
                "reacq_threshold": float(reacq.threshold),
                "reacq_confirm_frames": max(1, int(reacq.confirm_frames)),
                "occ_siam_reacq_threshold": float(reacq.occ_siam_threshold),
            }
        )

    if subsystems.gmc_prior is not None:
        gmc = subsystems.gmc_prior
        ram_overrides.update(
            {
                "gmc_prior_enabled": bool(gmc.enabled),
                "gmc_prior_require_reliable_h": bool(gmc.require_reliable_h),
                "gmc_prior_max_translation_frac": float(gmc.max_translation_frac),
                "gmc_prior_min_scale": float(gmc.min_scale),
                "gmc_prior_max_scale": float(gmc.max_scale),
                "gmc_prior_max_rotation_deg": float(gmc.max_rotation_deg),
                "gmc_prior_max_corner_displacement_frac": float(
                    gmc.max_corner_displacement_frac
                ),
            }
        )

    return ram_overrides, exp_overrides
