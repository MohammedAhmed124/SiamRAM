#!/usr/bin/env bash
# Convert the data/<ds>/<vid>/img symlinks into real directories by moving the
# underlying root-owned data_imgs tree into place, then delete the empty
# data_imgs root.
#
# This is a one-shot cleanup. After running it, data/ is fully self-contained
# and data_imgs/ no longer exists. The repo's code and CSVs already expect the
# new layout — only the on-disk relocation needs root.
#
# Run from the repo root:
#     sudo bash scripts/consolidate_data_imgs.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_IMGS="$REPO_ROOT/data_imgs"
DATA="$REPO_ROOT/data"

if [ ! -d "$DATA_IMGS" ]; then
    echo "[consolidate] $DATA_IMGS not present — nothing to do."
    exit 0
fi

if [ "$(id -u)" -ne 0 ]; then
    echo "[consolidate] must run as root (sudo bash $0)"
    exit 1
fi

REPO_OWNER="$(stat -c '%U' "$DATA")"
REPO_GROUP="$(stat -c '%G' "$DATA")"

moved=0
errored=0
shopt -s nullglob
for img_dir in "$DATA_IMGS"/*/*/img; do
    rel="${img_dir#$DATA_IMGS/}"
    target_parent="$DATA/${rel%/img}"
    target="$target_parent/img"

    if [ ! -d "$target_parent" ]; then
        mkdir -p "$target_parent"
        chown "$REPO_OWNER:$REPO_GROUP" "$target_parent"
    fi

    if [ -L "$target" ]; then
        rm "$target"
    elif [ -d "$target" ] && [ ! -L "$target" ]; then
        echo "[consolidate] WARNING: $target already exists as real directory; leaving in place" >&2
        errored=$((errored + 1))
        continue
    fi

    mv "$img_dir" "$target"
    chown -R "$REPO_OWNER:$REPO_GROUP" "$target"
    moved=$((moved + 1))
done
shopt -u nullglob

# Remove now-empty dataset directories inside data_imgs/ and the root itself.
find "$DATA_IMGS" -mindepth 1 -depth -type d -empty -delete || true
if [ -z "$(ls -A "$DATA_IMGS" 2>/dev/null)" ]; then
    rmdir "$DATA_IMGS"
    echo "[consolidate] removed empty $DATA_IMGS"
else
    echo "[consolidate] WARNING: $DATA_IMGS still has contents — not removing." >&2
fi

echo "[consolidate] moved=$moved errored=$errored"
