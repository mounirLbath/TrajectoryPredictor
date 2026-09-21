#!/usr/bin/env bash
# Downloads a subset of Argoverse 2 Motion Forecasting (CC BY-NC-SA 4.0): the first N scenarios per split.
# Usage: ./download_data.sh [split:count ...]     default: train:30000 val:2000
set -euo pipefail

BUCKET=s3://argoverse/datasets/av2/motion-forecasting
DEST=$(dirname "$0")/data/av2

command -v s5cmd >/dev/null || { echo "s5cmd not found: brew install s5cmd"; exit 1; }

specs=("$@")
[ ${#specs[@]} -gt 0 ] || specs=(train:30000 val:2000)

for spec in "${specs[@]}"; do
    split=${spec%%:*}
    count=${spec##*:}
    s5cmd --no-sign-request ls "$BUCKET/$split/" \
        | awk -v n="$count" -v b="$BUCKET/$split" -v d="$DEST/$split" '/DIR/ && ++i <= n {print "sync", b "/" $NF "*", d "/" $NF}' \
        | s5cmd --no-sign-request run
    echo "$split: $count scenarios"
done
