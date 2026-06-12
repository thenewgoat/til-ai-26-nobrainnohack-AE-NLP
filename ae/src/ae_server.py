"""Runs the AE server."""

# Unless you want to do something special with the server, you shouldn't need
# to change anything in this file.


import json
import os

from ae_manager import AEManager, HybridAEManager, NeuralAEManager
from fastapi import FastAPI, Request

app = FastAPI()

# AE_MODE selects the deployed agent: "scripted" (rule cascade — known-strong
# baseline) or "neural" (trained BC clone served via ONNX). Default scripted.
_mode = os.environ.get("AE_MODE", "scripted").lower()
if _mode == "neural":
    manager = NeuralAEManager()
elif _mode == "hybrid":
    manager = HybridAEManager()
elif _mode == "scripted":
    manager = AEManager()
else:
    raise ValueError(f"AE_MODE={_mode!r} must be 'scripted', 'neural', or 'hybrid'")


@app.post("/ae")
async def ae(request: Request) -> dict[str, list[dict[str, int]]]:
    """Feeds an observation into the AE model.

    Returns action taken given current observation (int)
    """

    # Read the raw body first so a malformed request is fully diagnosable
    # in the logs (caller, headers, byte-level body) instead of crashing
    # opaquely inside FastAPI's automatic JSON parser.
    raw = await request.body()
    try:
        input_json = json.loads(raw)
        instances = input_json["instances"]
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        client = f"{request.client.host}:{request.client.port}" if request.client else "?"
        print(f"BAD /ae request from {client}: {type(e).__name__}: {e}",
              flush=True)
        print(f"  headers: {dict(request.headers)}", flush=True)
        print(f"  body ({len(raw)} bytes): {raw!r}", flush=True)
        raise

    predictions = []
    # each is a dict with one key "observation" and the value as a dictionary observation
    for instance in instances:
        observation = instance["observation"]
        # reset environment on a new round
        # You will have to do your own internal counting and reset your own system between rounds!
        # if observation["step"] == 0:
            # do internal resetting here
        predictions.append({"action": manager.ae(observation)})
    return {"predictions": predictions}


# ------------------------------ RESET REMOVED ------------------------------


@app.get("/health")
def health() -> dict[str, str]:
    """Health check function for your model."""
    return {"message": "health ok"}
