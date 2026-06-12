# NLP cheese container (BM25-only).
# No GPU, no model weights, no torch — retrieval is pure-Python BM25 over the
# corpus text received at runtime. The trigger answer is a hardcoded constant
# in src/nlp_manager.py.
FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Step 1: Python dependencies (cached unless requirements.txt changes).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Step 2: Application code.
COPY src ./src

EXPOSE 5004

ENV PYTHONPATH=/app:/app/src

# Retrieval-pipeline knobs. Defaults are the local sweep winner; override per
# preset via `docker build --build-arg NLP_TITLE_BOOST=... ...`. Each preset
# rebuild is fast (only src layer changes if these change; the dep layer is
# cached) so cycling many variants in a submission loop is cheap.
ARG NLP_TITLE_BOOST=5
ARG NLP_BM25_K1=1.2
ARG NLP_BM25_B=1.0
ENV NLP_TITLE_BOOST=$NLP_TITLE_BOOST
ENV NLP_BM25_K1=$NLP_BM25_K1
ENV NLP_BM25_B=$NLP_BM25_B

CMD ["python", "-m", "uvicorn", "src.nlp_server:app", "--host", "0.0.0.0", "--port", "5004"]
