FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MOD_KEY_DATA_DIR=/tmp

WORKDIR /workspace

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt pyproject.toml README.md ./
COPY modulation_key_estimator ./modulation_key_estimator

RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir --no-deps -e .

EXPOSE 8000

CMD ["uvicorn", "modulation_key_estimator.api:app", "--host", "0.0.0.0", "--port", "8000"]
