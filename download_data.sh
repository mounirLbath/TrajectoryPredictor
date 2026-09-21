#!/usr/bin/env bash
# Downloads Argoverse 2 Motion Forecasting (CC BY-NC-SA 4.0) from its public S3 bucket.
# Usage: ./download_data.sh [val|train|test ...]     default: val train
set -euo pipefail

BUCKET=s3://argoverse/datasets/av2/motion-forecasting
DEST=$(dirname "$0")/data/av2

command -v s5cmd >/dev/null || { echo "s5cmd not found: brew install s5cmd"; exit 1; }

for split in "${@:-val train}"; do
    mkdir -p "$DEST/$split"
    echo "downloading $split ..."
    s5cmd --no-sign-request --numworkers 32 sync "$BUCKET/$split/*" "$DEST/$split/"
    echo "$split: $(find "$DEST/$split" -mindepth 1 -maxdepth 1 -type d | wc -l) scenarios"
done
