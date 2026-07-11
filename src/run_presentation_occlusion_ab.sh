#!/usr/bin/env bash
#
# Render a presentation video for ONE sequence twice: once with occlusion
# recovery ON and once with it OFF, so the two runs can be compared side by side.
#
# Each run uses a dedicated inference config:
#   ON  -> inference_config.yaml
#   OFF -> inference_config_no_occlusion.yaml
#          (differs by enter_occlusion_on_loss: False and disable_camera_motion: True)
#
# Usage
# -----
#   src/run_presentation_occlusion_ab.sh --key "dataset1/basketball"
#   src/run_presentation_occlusion_ab.sh --key "dataset1/basketball" \
#       --split public_lb --outputs-base outputs_presentation
#   src/run_presentation_occlusion_ab.sh --key "dataset1/basketball" \
#       --config-on inference_config.yaml --config-off inference_config_no_occlusion.yaml
#
# Anything after a lone `--` is passed straight through to the python script
# (e.g. `-- --fast`).
#
# Outputs land under:
#   <outputs-base>/occlusion_on/<dataset>/<video>/...
#   <outputs-base>/occlusion_off/<dataset>/<video>/...
#
set -euo pipefail

# --- resolve repo root (this script lives in <root>/src) --------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG_DIR="$REPO_ROOT/src/config"

# --- defaults ---------------------------------------------------------------
KEY=""
CONFIG_ON="inference_config.yaml"
CONFIG_OFF="inference_config_no_occlusion.yaml"
SPLIT="public_lb"
OUTPUTS_BASE="outputs_presentation"
EXTRA_ARGS=()

# --- parse args -------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --key)          KEY="$2"; shift 2 ;;
    --config-on)    CONFIG_ON="$2"; shift 2 ;;
    --config-off)   CONFIG_OFF="$2"; shift 2 ;;
    --split)        SPLIT="$2"; shift 2 ;;
    --outputs-base) OUTPUTS_BASE="$2"; shift 2 ;;
    --)             shift; EXTRA_ARGS+=("$@"); break ;;   # pass-through to the python script
    -h|--help)
      sed -n '2,22p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *)
      echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$KEY" ]]; then
  echo "error: --key is required (e.g. --key \"dataset1/basketball\")" >&2
  exit 2
fi

# --- resolve a config path (bare name -> src/config) ------------------------
resolve_config() {
  local c="$1"
  if [[ "$c" = /* ]]; then echo "$c"; else echo "$CONFIG_DIR/$c"; fi
}
BASE_ON="$(resolve_config "$CONFIG_ON")"
BASE_OFF="$(resolve_config "$CONFIG_OFF")"
for f in "$BASE_ON" "$BASE_OFF"; do
  if [[ ! -f "$f" ]]; then
    echo "error: config not found: $f" >&2
    exit 2
  fi
done

PY="$REPO_ROOT/src/run_inference_video_presentation.py"

run_one() {
  local cfg="$1" outdir="$2" label="$3"
  echo ""
  echo "=================================================================="
  echo ">>> OCCLUSION $label  |  key=$KEY  |  config=$(basename "$cfg")"
  echo ">>> output: $OUTPUTS_BASE/$outdir"
  echo "=================================================================="
  python "$PY" \
    --config "$cfg" \
    --split "$SPLIT" \
    --key "$KEY" \
    --outputs-dir "$OUTPUTS_BASE/$outdir" \
    "${EXTRA_ARGS[@]}"
}

cd "$REPO_ROOT"
run_one "$BASE_ON"  "occlusion_on"  "ON"
run_one "$BASE_OFF" "occlusion_off" "OFF"

echo ""
echo "Done. Compare:"
echo "  $OUTPUTS_BASE/occlusion_on"
echo "  $OUTPUTS_BASE/occlusion_off"
