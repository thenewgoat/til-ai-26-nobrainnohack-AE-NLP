#!/usr/bin/env bash
# Wait for the fixed pilot to finish, gate on whether post_handover_return improved,
# then launch the 3000-update full run. Detached via nohup.
set -u
cd /home/jupyter/til-ai-26/ae
PILOT_PID=78229
mkdir -p runs/full
LOG=runs/full/queue.log
echo "$(date) waiting for pilot PID $PILOT_PID ..." > "$LOG"

while kill -0 "$PILOT_PID" 2>/dev/null; do sleep 30; done
echo "$(date) pilot process exited" >> "$LOG"

GATE=$(PYTHONPATH=src:src/training ../til-26-ae/.venv/bin/python - <<'PY'
import json
try:
    rows = [json.loads(l) for l in open("runs/pilot_fix_s0/metrics.jsonl")]
except Exception:
    print("SKIP no-metrics"); raise SystemExit
upd = [m for m in rows if "post_handover_return" in m and "actor_lr" in m]
if len(upd) < 40:
    print("SKIP too-short"); raise SystemExit
early = sum(m["post_handover_return"] for m in upd[:30]) / 30
late  = sum(m["post_handover_return"] for m in upd[-30:]) / 30
e_ent = sum(m.get("entropy", 0) for m in upd[:30]) / 30
l_ent = sum(m.get("entropy", 0) for m in upd[-30:]) / 30
ok = (late - early) > 10 or late > -70
print(f"{'PASS' if ok else 'FAIL'} early_ret={early:.1f} late_ret={late:.1f} ent {e_ent:.2f}->{l_ent:.2f}")
PY
)
echo "$(date) gate: $GATE" >> "$LOG"

if echo "$GATE" | grep -q PASS; then
  echo "$(date) launching FULL run (3000 updates)" >> "$LOG"
  PYTHONPATH=src:src/training ../til-26-ae/.venv/bin/python src/training/run_hybrid.py \
    --updates 3000 --episodes-per-update 4 --rollout-workers 4 --device cuda \
    --lr 3e-4 --update-epochs 2 --critic-warmup 10 \
    --forward-bias 0.5 --anti-idle 0.10 --step-fallback 60 \
    --opponents balanced balanced_extreme_opening adaptive forager defender lean_rush \
    --checkpoint-dir runs/full --checkpoint-every 100 \
    --eval-every 250 --eval-seeds 16 --log runs/full/metrics.jsonl \
    --export-onnx runs/full/rl_actor.onnx --seed0 0 >> runs/full/run.log 2>&1
  echo "$(date) full run exited code $?" >> "$LOG"
else
  echo "$(date) gate did NOT pass -> full run NOT launched. Inspect runs/pilot_fix_s0." >> "$LOG"
fi
