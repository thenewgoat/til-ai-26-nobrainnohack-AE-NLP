#!/usr/bin/env bash
# Cycle through preset (title_boost, k1, b) configurations: for each one,
# rebuild the nlp container with build-args, then `til submit` it, then sleep.
#
# The Dockerfile reads NLP_TITLE_BOOST / NLP_BM25_K1 / NLP_BM25_B from ARGs
# so each preset bakes those values as ENV into a uniquely-tagged image.
# Tag convention: t<title>-k<k1*10>-b<b*10> (e.g. t5-k12-b10).
#
# Usage (run from any directory):
#   nohup bash nlp_cheese/submit_presets.sh > submit_presets.log 2>&1 &
#   tail -f submit_presets.log
# Stop with:  pkill -f submit_presets.sh
set -uo pipefail

: "${TIL_FOLDER:=/home/jupyter/til-ai-26}"
: "${TEAM_NAME:?must be set in the environment}"

# Each entry: "tag:title_boost:k1:b"
# Edit / reorder / add to taste. Local-set recall@3 from tune_bm25.py shown.
PRESETS=(
    "t5-k12-b10:5:1.2:1.0"      # 0.9830  current sweep winner
    "t5-k20-b10:5:2.0:1.0"      # 0.9819
    "t5-k20-b075:5:2.0:0.75"    # 0.9819
    "t5-k15-b075:5:1.5:0.75"    # 0.9819  title-only, rank_bm25 defaults
    "t5-k12-b05:5:1.2:0.5"      # 0.9773  less length normalization
    "t10-k12-b10:10:1.2:1.0"    # 0.9819  (untested but expected — title heavier)
    "t3-k15-b075:3:1.5:0.75"    # 0.9807  weaker title
    "t0-k15-b075:0:1.5:0.75"    # 0.9807  baseline plain BM25 (no title)
)

SLEEP_SEC="${SLEEP_SEC:-600}"   # 10 minutes between submissions

for entry in "${PRESETS[@]}"; do
    IFS=':' read -r tag tb k1 b <<<"$entry"
    image="${TEAM_NAME}-nlp:${tag}"
    ts=$(date '+%Y-%m-%d %H:%M:%S')
    echo
    echo "[$ts] === preset $tag : TITLE_BOOST=$tb  BM25_K1=$k1  BM25_B=$b ==="

    echo "[$ts] building $image"
    docker build \
        --build-arg "NLP_TITLE_BOOST=$tb" \
        --build-arg "NLP_BM25_K1=$k1" \
        --build-arg "NLP_BM25_B=$b" \
        -t "$image" \
        -f "$TIL_FOLDER/nlp/Dockerfile" \
        "$TIL_FOLDER/nlp"
    if [[ $? -ne 0 ]]; then
        echo "[$(date '+%H:%M:%S')] BUILD FAILED for $tag — skipping submit"
        sleep "$SLEEP_SEC"
        continue
    fi

    echo "[$(date '+%H:%M:%S')] submitting $tag"
    til submit nlp "$tag"
    if [[ $? -ne 0 ]]; then
        echo "[$(date '+%H:%M:%S')] SUBMIT FAILED for $tag"
    else
        echo "[$(date '+%H:%M:%S')] submitted $tag OK"
    fi

    echo "[$(date '+%H:%M:%S')] sleeping ${SLEEP_SEC}s before next preset"
    sleep "$SLEEP_SEC"
done

echo "[$(date '+%H:%M:%S')] all ${#PRESETS[@]} presets processed"
