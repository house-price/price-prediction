# syntax=docker/dockerfile:1
FROM python:3.12-slim

# Disable pyc writes + unbuffered logs (recommended in containers)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Copy and install the requirements first: this layer is cached as long as requirements is unchanged (faster builds)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Then copy source code and data
COPY app ./app
COPY data ./data
COPY train.py .

# Train at build time so the model is baked into the image (ready on start, reproducible, no train at runtime)
RUN python train.py

# Run as a non-root user (security best practice)
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

# Built-in health check so orchestrators can tell whether the instance is usable
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').getcode()==200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
