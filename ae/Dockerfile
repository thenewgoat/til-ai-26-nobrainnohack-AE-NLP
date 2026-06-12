# Dockerfile for building the AE image.


# The base image. Using a regular Python base because AE agents don't typically
# require GPU. This will speed up your evaluation.
FROM python:3.11-slim

# Configures settings for the image.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_ROOT_USER_ACTION=ignore
WORKDIR /workspace

# Installs your dependencies.
RUN pip install -U pip
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copies your source files. `.dockerignore` excludes src/training/ (the
# training pipeline) and other dev-only paths from the image — only the
# runtime modules (scripted/, ae_*, policy.py, features.py, *.json, *.pt,
# *.onnx) end up in /workspace.
COPY src .

# Which agent the server runs. Default "scripted" (proven rule cascade);
# "neural" loads policy.onnx via NeuralAEManager.
#   scripted — rule cascade (see AE_STRATEGY below for which one).
#   neural   — trained BC clone served via ONNX (ignores AE_STRATEGY).
#   hybrid   — scripted opener -> RL actor (ONNX) post-handover; needs AE_RL_ACTOR_PATH.
# Override per build: docker build --build-arg AE_MODE=neural ...
# Or per run:         docker run -e AE_MODE=neural ...
# Default: hybrid (scripted opener -> trained RL actor post-handover). The actor
# is baked at /workspace/rl_actor.onnx (see AE_RL_ACTOR_PATH below). Switch to
# scripted/neural with --build-arg AE_MODE=scripted (AE_STRATEGY then applies).
ARG AE_MODE=scripted
ENV AE_MODE=${AE_MODE}

# Which scripted strategy this image serves when AE_MODE=scripted. Override
# per build: docker build --build-arg AE_STRATEGY=<name> ...
# Must be a key in ae/src/scripted/strategies.py STRATEGIES — one of:
#   balanced, balanced_extreme, base_rusher, base_rusher_extreme,
#   collector, camper, forager, lean_rush, defender, adaptive.
# Or the standalone "greedy" agent (ae/src/scripted/greedy.py).
# Ignored when AE_MODE=neural.
ARG AE_STRATEGY=balanced_extreme_opening
ENV AE_STRATEGY=${AE_STRATEGY}

# When AE_MODE=hybrid, the server loads the trained RL actor from this baked
# .onnx (exported via training/export_onnx.py and COPYed in with `COPY src .`).
# A missing path is a FATAL startup error (no silent fallback).
ARG AE_RL_ACTOR_PATH=/workspace/rl_actor.onnx
ENV AE_RL_ACTOR_PATH=${AE_RL_ACTOR_PATH}

# Starts your model server.
CMD uvicorn ae_server:app --port 5005 --host 0.0.0.0
#CMD ["python", "-m", "uvicorn", "src.ae_server:app", "--host", "0.0.0.0", "--port", "5005"]
