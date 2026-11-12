# CPU image for the SoccerVision API + worker. Model weights and data are mounted, not baked in.
#
#   docker build -t soccervision-api .
#   docker run --rm -p 8000:8000 -v "$PWD/data:/app/data" -v "$PWD/models:/app/models" soccervision-api
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    SOCCERVISION_DATA_DIR=/app/data \
    SOCCERVISION_WEIGHTS_DIR=/app/models \
    SOCCERVISION_DEFAULT_DEVICE=cpu \
    SOCCERVISION_CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:3000

WORKDIR /app

# CPU-only PyTorch wheels avoid pulling multi-GB CUDA libraries.
RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.15,<2.16" "torchvision>=0.29,<0.30"

COPY pyproject.toml README.md LICENSE ./
COPY packages ./packages
COPY apps/api ./apps/api
RUN pip install ".[ml,api]"

RUN useradd --create-home --uid 1000 app && mkdir -p /app/data /app/models && chown -R app /app
USER app

EXPOSE 8000
CMD ["uvicorn", "soccervision_api.main:app", "--host", "0.0.0.0", "--port", "8000"]
