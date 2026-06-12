#!/usr/bin/env bash
# probe_sweep.sh — Phase 1 of the BenBot probe campaign.
#
# Builds + submits a basis of archetypal scripted AE strategies on a fixed
# cadence, so the leaderboard score deltas tell us what kind of game the
# BenBots play (collector wins => BenBots beat each other up; camper wins
# => they can't dislodge entrenched positions; base_rusher_extreme wins
# => they have weak bases; etc.).
#
# Usage:
#   ./probe_sweep.sh                       # build all + submit all
#   ./probe_sweep.sh build                 # build images only, don't submit
#   ./probe_sweep.sh submit <ts>           # submit pre-built images for <ts>
#
# Environment:
#   INTERVAL    seconds between submissions (default 600 = 10 min)
#   STRATS      space-separated override list of strategies
#   ANCHOR      set to "1" to bookend the sweep with a BXO control submit
#               on each end (anchors the noise floor, costs 2 extra slots)
#
# Output:
#   probe_sweep_<ts>.log next to the script, one line per action.

set -u

DEFAULT_STRATS="collector camper base_rusher_extreme defender forager"
STRATS_RAW=${STRATS:-$DEFAULT_STRATS}
read -r -a STRATEGIES <<< "$STRATS_RAW"

INTERVAL=${INTERVAL:-600}
ANCHOR=${ANCHOR:-0}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AE_DIR="$SCRIPT_DIR"
export TIL_FOLDER="${TIL_FOLDER:-$(cd "$AE_DIR/.." && pwd)}"

MODE=${1:-all}
case "$MODE" in
    all|build) TS=$(date +%Y%m%d-%H%M) ;;
    submit)
        TS=${2:-}
        if [[ -z "$TS" ]]; then
            echo "submit mode needs a timestamp: $0 submit <YYYYMMDD-HHMM>" >&2
            exit 2
        fi
        ;;
    *)
        echo "usage: $0 [all|build|submit <ts>]" >&2
        exit 2
        ;;
esac

LOG="$SCRIPT_DIR/probe_sweep_${TS}.log"

log() { echo "[$(date -Iseconds)] $*" | tee -a "$LOG"; }

# Build one image. Echoes the bare tag on success, returns non-zero on failure.
build_one() {
    local strat=$1
    local tag="probe-${strat}-${TS}"
    log "build start: til-ae:${tag} (AE_STRATEGY=${strat})"
    if docker build \
            --build-arg AE_MODE=scripted \
            --build-arg AE_STRATEGY="${strat}" \
            -t "til-ae:${tag}" \
            "$AE_DIR" >>"$LOG" 2>&1; then
        log "build OK:    til-ae:${tag}"
        echo "$tag"
        return 0
    else
        log "BUILD FAIL:  til-ae:${tag} (see $LOG)"
        return 1
    fi
}

# Submit one tag. Always returns 0 (failures are logged, sweep keeps going).
submit_one() {
    local tag=$1
    log "submit start: ${tag}"
    if til submit ae "${tag}" >>"$LOG" 2>&1; then
        log "submit OK:    ${tag}"
    else
        local rc=$?
        log "SUBMIT FAIL:  ${tag} (rc=$rc)"
    fi
}

count=0
trap 'log "interrupted after $count submission(s)"; exit 130' INT TERM

# Build the queue. Optional anchor BXO at start + end.
QUEUE=()
if [[ "$ANCHOR" == "1" ]]; then
    QUEUE+=("balanced_extreme_opening")
fi
QUEUE+=("${STRATEGIES[@]}")
if [[ "$ANCHOR" == "1" ]]; then
    QUEUE+=("balanced_extreme_opening")
fi

log "=== probe sweep: ts=${TS} mode=${MODE} interval=${INTERVAL}s ==="
log "queue: ${QUEUE[*]}"

TAGS=()

# Build phase (skipped in submit mode).
if [[ "$MODE" == "all" || "$MODE" == "build" ]]; then
    log "--- build phase: ${#QUEUE[@]} images ---"
    seen_bxo=0
    for strat in "${QUEUE[@]}"; do
        # Anchor BXO is built once and reused for both anchor submits.
        if [[ "$strat" == "balanced_extreme_opening" && "$seen_bxo" == "1" ]]; then
            TAGS+=("probe-${strat}-${TS}")
            continue
        fi
        if tag=$(build_one "$strat"); then
            TAGS+=("$tag")
            [[ "$strat" == "balanced_extreme_opening" ]] && seen_bxo=1
        fi
    done
    log "built ${#TAGS[@]}/${#QUEUE[@]} images"
fi

if [[ "$MODE" == "build" ]]; then
    log "build-only mode: done. tags:"
    for t in "${TAGS[@]}"; do log "  $t"; done
    exit 0
fi

# In submit mode we did no building -- recover tag list from the queue.
if [[ "$MODE" == "submit" ]]; then
    for strat in "${QUEUE[@]}"; do
        TAGS+=("probe-${strat}-${TS}")
    done
    log "submit-only mode: will submit ${#TAGS[@]} pre-built tags"
fi

# Submit phase.
log "--- submit phase: ${#TAGS[@]} tags, ${INTERVAL}s between ---"
for i in "${!TAGS[@]}"; do
    submit_one "${TAGS[$i]}"
    count=$((count + 1))
    if (( i < ${#TAGS[@]} - 1 )); then
        log "sleeping ${INTERVAL}s"
        sleep "$INTERVAL"
    fi
done

log "=== sweep complete: ${count} submission(s) ==="
log "Correlation table (record leaderboard scores against these tags):"
for tag in "${TAGS[@]}"; do
    log "  $tag"
done
