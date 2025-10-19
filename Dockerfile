# syntax=docker/dockerfile:1
FROM python:3.11-slim

# --- Variables de entorno (consolidadas) ---
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=8000 \
    KERAS_HOME=/app/.keras \
    TF_CPP_MIN_LOG_LEVEL=2 \
    ARGOS_TRANSLATE_PACKAGE_DIR=/app/argos_data \
    UVICORN_WORKERS=1

WORKDIR /app

# --- Instalación de dependencias del sistema (minimales) ---
# Nota: mantenemos --no-install-recommends y limpiamos cache de apt para imágenes más pequeñas.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      pkg-config \
      wget \
      ca-certificates \
      libjpeg-dev \
      zlib1g-dev \
      libpng-dev \
      libtiff5-dev \
      libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Crear usuario no-root (recomendado)
RUN groupadd -r app && useradd -r -g app app \
    && mkdir -p /app /app/.keras /app/argos_data \
    && chown -R app:app /app

# --- Copiar solo requirements primero para aprovechar cache de Docker ---
COPY requirements.txt /app/requirements.txt
USER app

# Upgrade pip y preinstalar numpy compatible para evitar conflictos con TF
# (siempre pincha la versión en requirements.txt para reproducibilidad)
RUN python -m pip install --upgrade pip setuptools wheel \
 && python -m pip install --no-cache-dir "numpy>=1.26.0,<1.27"

# --- Instalar dependencias Python ---
# Si usas BuildKit, puedes acelerar con: --mount=type=cache,target=/root/.cache/pip
# Ejemplo: RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt
RUN python -m pip install --no-cache-dir -r /app/requirements.txt

# --- Copiar el código fuente (al final para aprovechar cache) ---
COPY --chown=app:app . /app

# Mantener carpetas de cache/mounts accesibles (volúmenes recomendados)
RUN mkdir -p /app/.keras /app/argos_data && chown -R app:app /app/.keras /app/argos_data

EXPOSE ${PORT}

# --- CMD (produce uvicorn). Para producción considera usar gunicorn + uvicorn workers ---
# Puedes ajustar UVICORN_WORKERS vía ENV o variable en tiempo de ejecución.
# Ejemplo local: docker run -e UVICORN_WORKERS=3 ...
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT} --loop asyncio --workers ${UVICORN_WORKERS}"]
