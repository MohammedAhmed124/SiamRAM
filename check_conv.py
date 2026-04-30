"""
convert_ckpt_fp16.py
====================
Mixed-precision checkpoint conversion that mirrors exactly how TRTSiamABCNet
uses each layer group at runtime:

    Layer group                  Runtime precision    Stored as
    ──────────────────────────── ────────────────── ──────────
    module.encoder.*             FP16 (TRT engine)  float16
    module.neck.*                FP16 (TRT engine)  float16
    module.polarized_self_att.*  FP32 (.float())    float32   ← kept
    module.attention_neck.*      FP32 (.float())    float32   ← kept
    module.connect_model.*       FP32 (.float())    float32   ← kept

Result: ~42 MB → ~24 MB  (encoder+neck halved, sensitive heads untouched)

Non-float tensors (int64 step counters, bool masks, etc.) are never touched
regardless of which group they belong to.

Usage
-----
    python convert_ckpt_fp16.py

Produces:
    head_epoch_000_mixed.pth   (same directory as the original)
"""

import os
import torch

# ── config ────────────────────────────────────────────────────────────────────
INPUT_PATH  = "/home/moha/AIC-for submission/SiamRAM/checkpoints/head_epoch_000.pth"
OUTPUT_PATH = "/home/moha/AIC-for submission/SiamRAM/checkpoints/head_epoch_000_mixed.pth"
# ─────────────────────────────────────────────────────────────────────────────

# These prefixes run at FP32 in TRTSiamABCNet — keep them float32.
# Every key not matching any of these goes to FP16.
FP32_PREFIXES = (
    "module.polarized_self_attention.",
    "module.attention_neck.",
    "module.connect_model.",
)


def _target_dtype(key: str) -> torch.dtype:
    """
    Decide the target dtype for a single checkpoint key.

    Returns float32 if the key belongs to the attention or connect_model
    groups (they run FP32 at runtime), float16 for everything else
    (encoder + neck, which run inside the FP16 TRT engine).
    """
    for prefix in FP32_PREFIXES:
        if key.startswith(prefix):
            return torch.float32
    return torch.float16


def convert(src: str, dst: str) -> None:
    print(f"Loading  : {src}")
    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    original_size_mb = os.path.getsize(src) / 1e6
    print(f"File size (before): {original_size_mb:.1f} MB\n")

    # ── per-group accounting ──────────────────────────────────────────────────
    stats = {
        "encoder":                  {"keys": 0, "mb_before": 0.0, "mb_after": 0.0},
        "neck":                     {"keys": 0, "mb_before": 0.0, "mb_after": 0.0},
        "polarized_self_attention": {"keys": 0, "mb_before": 0.0, "mb_after": 0.0},
        "attention_neck":           {"keys": 0, "mb_before": 0.0, "mb_after": 0.0},
        "connect_model":            {"keys": 0, "mb_before": 0.0, "mb_after": 0.0},
        "other":                    {"keys": 0, "mb_before": 0.0, "mb_after": 0.0},
    }

    def _group(key):
        stripped = key.replace("module.", "", 1)
        for g in stats:
            if stripped.startswith(g):
                return g
        return "other"

    # ── cast ──────────────────────────────────────────────────────────────────
    converted = {}
    for key, value in ckpt.items():
        if not isinstance(value, torch.Tensor) or not value.is_floating_point():
            # int / bool / long tensors (step counters, masks) → untouched
            converted[key] = value
            continue

        g = _group(key)
        stats[g]["mb_before"] += value.nbytes / 1e6
        stats[g]["keys"] += 1

        target = _target_dtype(key)
        converted[key] = value.to(target)
        stats[g]["mb_after"] += converted[key].nbytes / 1e6

    # ── report ────────────────────────────────────────────────────────────────
    print(f"  {'Group':<30}  {'Keys':>5}  {'Before':>8}  {'After':>8}  Action")
    print(f"  {'-'*30}  {'-'*5}  {'-'*8}  {'-'*8}  {'-'*12}")
    for g, s in stats.items():
        if s["keys"] == 0:
            continue
        action = "-> FP16" if g in ("encoder", "neck") else "   FP32 (kept)"
        print(f"  {g:<30}  {s['keys']:>5}  {s['mb_before']:>6.1f} MB  {s['mb_after']:>6.1f} MB  {action}")

    print()
    print(f"Saving   : {dst}")
    torch.save(converted, dst)

    new_size_mb = os.path.getsize(dst) / 1e6
    print(f"\nFile size (before): {original_size_mb:.1f} MB")
    print(f"File size (after) : {new_size_mb:.1f} MB")
    print(f"Reduction         : {100*(1 - new_size_mb/original_size_mb):.1f}%")
    print("Done.")


if __name__ == "__main__":
    convert(INPUT_PATH, OUTPUT_PATH)