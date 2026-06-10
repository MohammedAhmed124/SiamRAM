"""
Typed config helpers for SiamRAM tracker options.

This module supports three config layouts:
1) Modern nested groups under `ram_tracker` (preferred).
2) Legacy flat `ram_tracker` keys.
3) Compatibility blocks (`ram_tracker_subsystems`, `ram_tracker_experiment`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
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
class SpikeSettleConfig:
    ratio: float = 0.70
    abs_norm_max: float = 0.30
    frames: int = 1
    from_spike_frac: float = 0.55


@dataclass
class SpikeAnchorConfig:
    history_window: int = 12
    trigger_norm: float = 0.90
    update_norm_max: float = 0.60


@dataclass
class SpikeRejectConfig:
    enabled: bool = True
    min_frames: int = 3
    min_score: float = 0.50
    history_window: int = 40
    min_history: int = 5
    ratio: float = 2.5
    watch_max_frames: int = 8
    settle: SpikeSettleConfig = field(default_factory=SpikeSettleConfig)
    anchor: SpikeAnchorConfig = field(default_factory=SpikeAnchorConfig)


@dataclass
class DistractorModeConfig:
    jump_mode_frames: int = 12
    jump_penalty_enabled: bool = True
    jump_penalty_weight: float = 0.25
    jump_penalty_sim_floor: float = 0.55
    jump_penalty_bank_topk: int = 5
    min_similarity: float = 0.70
    selected_min_similarity: Any = 0.70
    selected_min_similarity_auto_min: float = 0.70
    selected_min_similarity_auto_max: float = 0.98
    selected_min_similarity_auto_delta: float = 0.03
    selected_min_similarity_auto_ema_alpha: float = 0.2
    selected_min_similarity_auto_n_frames: int = 10
    selected_min_similarity_auto_warmup: float = 0.98
    selected_min_similarity_auto_min_samples: int = 5
    selected_min_similarity_auto_use_distractor_bank: bool = True
    selected_min_similarity_auto_distractor_margin: float = 0.03
    yolo_topk: int = 8
    history_limit: int = 80
    use_tracker_mapping: bool = True
    roi_expand: float = 1.20
    roi_min_side: int = 32
    compare_mode: str = "ram"
    update_memory: bool = False
    exit_reinit_enabled: bool = True
    exit_stable_frames: int = 6
    exit_same_iou: float = 0.60
    anchor_ekf_enabled: bool = True
    anchor_uncertainty_roi_scale: float = 0.0
    anchor_uncertainty_roi_cap: float = 96.0
    mahalanobis_gate_enabled: bool = True
    mahalanobis_threshold: float = 9.21
    mahalanobis_meas_var: float = 25.0
    switch_margin: float = 0.08
    ambiguity_hold_frames: int = 2
    selected_below_gate_hold_frames: int = 0
    selected_below_gate_force_occlusion: bool = False
    reentry_cooldown_frames: int = 12
    post_exit_memory_freeze_frames: int = 8
    post_exit_template_freeze_frames: int = 8
    behavior_mode: str = "standard"
    prebank_enabled: bool = False
    prebank_stride: int = 10
    prebank_maxlen: int = 15
    prebank_yolo_topk: int = 5
    prebank_target_iou_max: float = 0.30
    prebank_materialize_immediately: bool = False
    overlap_motion_lock_enabled: bool = True
    overlap_iou_enter: float = 0.35
    overlap_iou_exit: float = 0.15
    overlap_clear_frames: int = 3
    overlap_lock_max_frames: int = 24
    focus_dist_penalty_enabled: bool = True
    focus_dist_penalty_weight: float = 0.35
    focus_dist_soft_radius: float = 0.60
    focus_dist_hard_radius: float = 2.00
    force_occlusion_on_commit: bool = False
    drm_lam_app: float = 1.0
    drm_lam_iou: float = 0.25
    drm_lam_dist: float = 1.0
    drm_gamma: float = 2.0
    drm_dist_sigma_factor: float = 2.5
    drm_bank_topk: int = 5


@dataclass
class GmcPriorConfig:
    enabled: bool = True
    require_reliable_h: bool = True
    skip_in_distractor_mode: bool = True
    max_translation_frac: float = 0.25
    min_scale: float = 0.70
    max_scale: float = 1.40
    max_rotation_deg: float = 25.0
    max_corner_displacement_frac: float = 0.25


@dataclass
class DrmIntrospectionConfig:
    """Introspection-based DRM anchor update (arXiv:2411.17576, Sec. 3.2.2).

    Master switch ``enabled`` defaults to False so the legacy DRM write path in
    ``AppearanceMemory`` is byte-for-byte unchanged until flipped on. When on, a
    DRM distractor-context anchor is written when the response map reveals a
    competing secondary peak (divergence bbox-area ratio < ``theta_anc``) during
    reliable tracking (score > ``theta_iou`` and target area within ``theta_area``
    of the median over the last ``theta_M`` frames), no more often than every
    ``delta`` frames, and only when the target is present.
    """

    enabled: bool = False
    theta_anc: float = 0.7
    theta_iou: float = 0.8
    theta_area: float = 0.2
    theta_M: int = 10
    delta: int = 5
    secondary_min_ratio: float = 0.5
    # Scale the divergence box + non-max suppression to the target's grid-cell
    # extent instead of a fixed 3x3 box, de-degenerating theta_anc on the coarse
    # 16x16 response map. True = scale-aware; False = legacy fixed 3x3.
    scale_aware_divergence: bool = True
    # Write the detected secondary-peak (distractor) descriptor into the NEGATIVE
    # distractor bank during reliable tracking, so the gamma-weighted suppression
    # term can penalize that look-alike. Inert unless a recovery gamma > 0.
    distractor_bank_enabled: bool = False


@dataclass
class FrameDynamicsConfig:
    """Frame-difference motion-saliency input augmentation (arXiv:2505.04917).

    Master switch ``enabled`` defaults to False so the search crop fed to the
    frozen SiamABC backbone is byte-for-byte unchanged until flipped on. When on,
    a short-term frame-difference motion-saliency signal (|x_t - x_{t-1}| and
    |x_t - x_{t-2}|) is computed between consecutive full frames and blended
    ADDITIVELY into the search region with weight ``blend_weight`` (scaled by
    ``scale`` and optionally clipped to ``clip`` when ``clip > 0`` to suppress
    outlier difference pixels). This deviates from the paper's channel
    concatenation (which assumed a from-scratch-trained 6-channel detector):
    SiamRAM cannot retrain the frozen 3-channel backbone, so the motion cue is
    injected via an additive 3-channel blend instead.

    By default the blend is LOW-WEIGHT (``blend_weight = 0.06``) and TINY-ONLY
    (``tiny_only = True``): it is applied only when the target is small (bbox area
    fraction ``<= tiny_area_fraction``, default ``0.001``), leaving
    normal/large-target crops clean.
    This confines the augmentation to the tiny-target regime where appearance
    collapses (per CST Anti-UAV) and the motion cue helps, avoiding corruption of
    the appearance signal the backbone matches on for the majority of frames.
    """

    enabled: bool = False
    blend_weight: float = 0.06
    scale: float = 1.0
    clip: float = -1.0
    tiny_only: bool = True
    tiny_area_fraction: float = 0.001


@dataclass
class ResearchTelemetryConfig:
    """Flag-gated diagnostics for the three research features.

    Single master switch ``research_telemetry_enabled`` (default False). When OFF
    there is zero behavioural change and effectively zero overhead. When ON the
    tracker maintains per-video and cumulative integer counters for the
    DRM-introspection feature and prints a compact summary line per
    video at the per-video reset point.
    """

    research_telemetry_enabled: bool = False


@dataclass
class TrackerSubsystemConfig:
    descriptor: Optional[DescriptorConfig] = None
    reacquisition: Optional[ReacquisitionConfig] = None
    spike_reject: Optional[SpikeRejectConfig] = None
    distractor_mode: Optional[DistractorModeConfig] = None
    gmc_prior: Optional[GmcPriorConfig] = None
    drm_introspection: Optional[DrmIntrospectionConfig] = None
    frame_dynamics: Optional[FrameDynamicsConfig] = None
    telemetry: Optional[ResearchTelemetryConfig] = None


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

    if subsystems.spike_reject is not None:
        spike = subsystems.spike_reject
        exp_overrides.update(
            {
                "spike_reject_enabled": bool(spike.enabled),
                "spike_reject_min_frames": int(spike.min_frames),
                "spike_reject_min_score": float(spike.min_score),
                "spike_reject_history_window": int(spike.history_window),
                "spike_reject_min_history": int(spike.min_history),
                "spike_reject_ratio": float(spike.ratio),
                "spike_reject_watch_max_frames": int(spike.watch_max_frames),
                "spike_reject_settle_ratio": float(spike.settle.ratio),
                "spike_reject_settle_abs_norm_max": float(spike.settle.abs_norm_max),
                "spike_reject_settle_frames": int(spike.settle.frames),
                "spike_reject_settle_from_spike_frac": float(
                    spike.settle.from_spike_frac
                ),
                "spike_anchor_history_window": int(spike.anchor.history_window),
                "spike_anchor_trigger_norm": float(spike.anchor.trigger_norm),
                "spike_anchor_update_norm_max": float(spike.anchor.update_norm_max),
            }
        )

    if subsystems.distractor_mode is not None:
        dis = subsystems.distractor_mode
        exp_overrides.update(
            {
                "jump_reject_distractor_mode_frames": int(dis.jump_mode_frames),
                "jump_reject_distractor_penalty_enabled": bool(
                    dis.jump_penalty_enabled
                ),
                "jump_reject_distractor_penalty_weight": float(dis.jump_penalty_weight),
                "jump_reject_distractor_penalty_sim_floor": float(
                    dis.jump_penalty_sim_floor
                ),
                "jump_reject_distractor_penalty_bank_topk": int(
                    dis.jump_penalty_bank_topk
                ),
                "distractor_mode_min_similarity": float(dis.min_similarity),
                "distractor_mode_selected_min_similarity": dis.selected_min_similarity,
                "distractor_mode_selected_min_similarity_auto_min": float(
                    dis.selected_min_similarity_auto_min
                ),
                "distractor_mode_selected_min_similarity_auto_max": float(
                    dis.selected_min_similarity_auto_max
                ),
                "distractor_mode_selected_min_similarity_auto_delta": float(
                    dis.selected_min_similarity_auto_delta
                ),
                "distractor_mode_selected_min_similarity_auto_ema_alpha": float(
                    dis.selected_min_similarity_auto_ema_alpha
                ),
                "distractor_mode_selected_min_similarity_auto_n_frames": int(
                    dis.selected_min_similarity_auto_n_frames
                ),
                "distractor_mode_selected_min_similarity_auto_warmup": float(
                    dis.selected_min_similarity_auto_warmup
                ),
                "distractor_mode_selected_min_similarity_auto_min_samples": int(
                    dis.selected_min_similarity_auto_min_samples
                ),
                "distractor_mode_selected_min_similarity_auto_use_distractor_bank": bool(
                    dis.selected_min_similarity_auto_use_distractor_bank
                ),
                "distractor_mode_selected_min_similarity_auto_distractor_margin": float(
                    dis.selected_min_similarity_auto_distractor_margin
                ),
                "distractor_mode_yolo_topk": int(dis.yolo_topk),
                "distractor_mode_history_limit": int(dis.history_limit),
                "distractor_mode_use_tracker_mapping": bool(dis.use_tracker_mapping),
                "distractor_mode_roi_expand": float(dis.roi_expand),
                "distractor_mode_roi_min_side": int(dis.roi_min_side),
                "distractor_compare_mode": str(dis.compare_mode),
                "distractor_mode_update_memory": bool(dis.update_memory),
                "distractor_mode_exit_reinit_enabled": bool(dis.exit_reinit_enabled),
                "distractor_mode_exit_stable_frames": int(dis.exit_stable_frames),
                "distractor_mode_exit_same_iou": float(dis.exit_same_iou),
                "distractor_mode_anchor_ekf_enabled": bool(dis.anchor_ekf_enabled),
                "distractor_mode_anchor_uncertainty_roi_scale": float(
                    dis.anchor_uncertainty_roi_scale
                ),
                "distractor_mode_anchor_uncertainty_roi_cap": float(
                    dis.anchor_uncertainty_roi_cap
                ),
                "distractor_mode_mahalanobis_gate_enabled": bool(
                    dis.mahalanobis_gate_enabled
                ),
                "distractor_mode_mahalanobis_threshold": float(
                    dis.mahalanobis_threshold
                ),
                "distractor_mode_mahalanobis_meas_var": float(
                    dis.mahalanobis_meas_var
                ),
                "distractor_mode_switch_margin": float(dis.switch_margin),
                "distractor_mode_ambiguity_hold_frames": int(
                    dis.ambiguity_hold_frames
                ),
                "distractor_mode_selected_below_gate_hold_frames": int(
                    dis.selected_below_gate_hold_frames
                ),
                "distractor_mode_selected_below_gate_force_occlusion": bool(
                    dis.selected_below_gate_force_occlusion
                ),
                "distractor_mode_reentry_cooldown_frames": int(
                    dis.reentry_cooldown_frames
                ),
                "distractor_mode_post_exit_memory_freeze_frames": int(
                    dis.post_exit_memory_freeze_frames
                ),
                "distractor_mode_post_exit_template_freeze_frames": int(
                    dis.post_exit_template_freeze_frames
                ),
                "distractor_mode_behavior_mode": str(dis.behavior_mode),
                "distractor_mode_prebank_enabled": bool(dis.prebank_enabled),
                "distractor_mode_prebank_stride": int(dis.prebank_stride),
                "distractor_mode_prebank_maxlen": int(dis.prebank_maxlen),
                "distractor_mode_prebank_yolo_topk": int(dis.prebank_yolo_topk),
                "distractor_mode_prebank_target_iou_max": float(
                    dis.prebank_target_iou_max
                ),
                "distractor_mode_prebank_materialize_immediately": bool(
                    dis.prebank_materialize_immediately
                ),
                "distractor_mode_overlap_motion_lock_enabled": bool(
                    dis.overlap_motion_lock_enabled
                ),
                "distractor_mode_overlap_iou_enter": float(dis.overlap_iou_enter),
                "distractor_mode_overlap_iou_exit": float(dis.overlap_iou_exit),
                "distractor_mode_overlap_clear_frames": int(dis.overlap_clear_frames),
                "distractor_mode_overlap_lock_max_frames": int(
                    dis.overlap_lock_max_frames
                ),
                "distractor_focus_dist_penalty_enabled": bool(
                    dis.focus_dist_penalty_enabled
                ),
                "distractor_focus_dist_penalty_weight": float(
                    dis.focus_dist_penalty_weight
                ),
                "distractor_focus_dist_soft_radius": float(dis.focus_dist_soft_radius),
                "distractor_focus_dist_hard_radius": float(dis.focus_dist_hard_radius),
                "jump_reject_force_occlusion": bool(dis.force_occlusion_on_commit),
                "distractor_drm_lam_app": float(dis.drm_lam_app),
                "distractor_drm_lam_iou": float(dis.drm_lam_iou),
                "distractor_drm_lam_dist": float(dis.drm_lam_dist),
                "distractor_drm_gamma": float(dis.drm_gamma),
                "distractor_drm_dist_sigma_factor": float(
                    dis.drm_dist_sigma_factor
                ),
                "distractor_drm_bank_topk": int(dis.drm_bank_topk),
            }
        )

    if subsystems.drm_introspection is not None:
        dri = subsystems.drm_introspection
        exp_overrides.update(
            {
                "drm_introspection_enabled": bool(dri.enabled),
                "drm_introspection_theta_anc": float(dri.theta_anc),
                "drm_introspection_theta_iou": float(dri.theta_iou),
                "drm_introspection_theta_area": float(dri.theta_area),
                "drm_introspection_theta_M": int(dri.theta_M),
                "drm_introspection_delta": int(dri.delta),
                "drm_introspection_secondary_min_ratio": float(
                    dri.secondary_min_ratio
                ),
                "drm_introspection_scale_aware_divergence": bool(
                    dri.scale_aware_divergence
                ),
                "drm_introspection_distractor_bank_enabled": bool(
                    dri.distractor_bank_enabled
                ),
            }
        )

    if subsystems.frame_dynamics is not None:
        fd = subsystems.frame_dynamics
        exp_overrides.update(
            {
                "frame_dynamics_enabled": bool(fd.enabled),
                "frame_dynamics_blend_weight": float(fd.blend_weight),
                "frame_dynamics_scale": float(fd.scale),
                "frame_dynamics_clip": float(fd.clip),
                "frame_dynamics_tiny_only": bool(fd.tiny_only),
                "frame_dynamics_tiny_area_fraction": float(fd.tiny_area_fraction),
            }
        )

    if subsystems.telemetry is not None:
        tel = subsystems.telemetry
        exp_overrides.update(
            {
                "research_telemetry_enabled": bool(tel.research_telemetry_enabled),
            }
        )

    if subsystems.gmc_prior is not None:
        gmc = subsystems.gmc_prior
        ram_overrides.update(
            {
                "gmc_prior_enabled": bool(gmc.enabled),
                "gmc_prior_require_reliable_h": bool(gmc.require_reliable_h),
                "gmc_prior_skip_in_distractor_mode": bool(
                    gmc.skip_in_distractor_mode
                ),
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
