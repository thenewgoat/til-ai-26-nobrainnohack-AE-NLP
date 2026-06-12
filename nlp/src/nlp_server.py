"""FastAPI server for the NLP RAG QA container.

Contract (current task spec, see test/test_nlp.py):
  POST /nlp
    - corpus load: {"instances": [{"documents": [{"id","document"}, ...]}]}
        -> {"predictions": [{"status": "loading"|"loaded"|"error"}]}
    - readiness poll: {"instances": [{"poll": "true"}]}
        -> {"predictions": [{"status": ...}]}
    - questions: {"instances": [{"question": ...}, ...]}
        -> {"predictions": [{"answer": str, "documents": [doc_id, ...]}, ...]}
  GET /health -> 200 {"message": "health ok"} once ready; 503 while loading

The model loads in a background thread so /health responds immediately. /health
is a real readiness check: 503 while loading, 503 (with error) if the load
failed, 200 only once the model is ready.
"""
import asyncio
import logging
import os
import sys
import threading
from typing import Optional

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from nlp_manager import NLPManager

app = FastAPI()
logger = logging.getLogger(__name__)

manager: Optional[NLPManager] = None
model_ready = threading.Event()
model_error: Optional[str] = None


def _init_manager() -> None:
    global manager, model_error
    try:
        manager = NLPManager()
    except Exception as e:
        logger.exception("Model load failed")
        model_error = str(e)
    finally:
        model_ready.set()


threading.Thread(target=_init_manager, daemon=True).start()


class _LoadState:
    def __init__(self) -> None:
        self.status = "idle"  # idle | loading | loaded | failed
        self.task: Optional[asyncio.Task] = None
        self.lock = asyncio.Lock()


load_state = _LoadState()


def _do_load(documents) -> bool:
    model_ready.wait()
    if manager is None:
        raise RuntimeError(f"Model failed to load: {model_error}")
    manager.load_corpus(documents)
    return manager.loaded


async def _load_task(documents) -> None:
    try:
        ok = await asyncio.to_thread(_do_load, documents)
        load_state.status = "loaded" if ok else "failed"
    except Exception:
        logger.exception("Corpus load failed")
        load_state.status = "failed"


def _status_payload() -> dict:
    """Wrap the load state in the {"status": ...} schema the scorer polls."""
    s = load_state.status
    if s == "idle":
        s = "loading"
    elif s == "failed":
        s = "error"
    return {"predictions": [{"status": s}]}


@app.post("/nlp")
async def nlp(request: Request) -> dict:
    inputs_json = await request.json()
    first = inputs_json["instances"][0]

    # Corpus-load request: documents arrive as {"id","document"} dicts.
    if first.get("documents") is not None:
        async with load_state.lock:
            if load_state.status == "idle":
                load_state.status = "loading"
                load_state.task = asyncio.create_task(
                    _load_task(first["documents"]))
        return _status_payload()

    # Readiness poll.
    if first.get("poll") is not None:
        return _status_payload()

    # QA batch: each prediction is {"answer": ..., "documents": [doc_id, ...]}.
    if manager is None:
        return {"predictions": [
            {"answer": "", "documents": []} for _ in inputs_json["instances"]
        ]}
    questions = [inst["question"] for inst in inputs_json["instances"]]
    predictions = await asyncio.to_thread(manager.qa_batch, questions)
    return {"predictions": predictions}


@app.get("/health")
def health():
    """Readiness check: 'healthy' only once the model is loaded and ready."""
    if model_error is not None:
        return JSONResponse(
            status_code=503,
            content={"message": f"model load failed: {model_error}"})
    if not model_ready.is_set():
        return JSONResponse(status_code=503, content={"message": "loading"})
    return {"message": "health ok"}
