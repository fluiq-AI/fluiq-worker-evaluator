FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # Cap native thread pools — torch/spaCy default to one pool per CPU core,
    # which is wasted RAM for our tiny single-message-at-a-time workload.
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false \
    # Bound glibc per-thread malloc arenas. Without this, every worker thread
    # that runs torch inference gets its own arena that grows and is never
    # returned to the OS, ratcheting RSS up to the container limit.
    MALLOC_ARENA_MAX=2

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "-m", "app"]
