"""Download all models the NLP container needs, into nlp_shizhen/models/.

Run this on the host (which has internet) before building the Docker image.
The Dockerfile then bakes models/ into the image so the container runs offline.

Models:
  - microsoft/Phi-4-mini-instruct   answer generator (native transformers Phi3)
  - BAAI/bge-large-en-v1.5          dense retrieval embedder
  - BAAI/bge-reranker-v2-m3         cross-encoder reranker
"""
import os

from sentence_transformers import CrossEncoder, SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

# models/ sits next to src/ regardless of the current working directory.
TARGET_DIR = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
)
os.makedirs(TARGET_DIR, exist_ok=True)
print(f"Downloading models into: {TARGET_DIR}", flush=True)

print("[1/3] Phi-4-mini-instruct ...", flush=True)
AutoTokenizer.from_pretrained("microsoft/Phi-4-mini-instruct", cache_dir=TARGET_DIR)
AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-4-mini-instruct", cache_dir=TARGET_DIR, attn_implementation="sdpa"
)

print("[2/3] bge-large-en-v1.5 (embedder) ...", flush=True)
SentenceTransformer("BAAI/bge-large-en-v1.5", cache_folder=TARGET_DIR)

print("[3/3] bge-reranker-v2-m3 (reranker) ...", flush=True)
CrossEncoder("BAAI/bge-reranker-v2-m3", cache_folder=TARGET_DIR)

print("\nDone. All three models are cached under models/.", flush=True)
