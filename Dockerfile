# syntax=docker/dockerfile:1
FROM python:3.11-slim

# --- Entornos y ajuste de paralelismo para CPU (2 vCPU) ---
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=8000 \
    KERAS_HOME=/app/.keras \
    TF_CPP_MIN_LOG_LEVEL=2 \
    # Limitar hilos para OpenBLAS, MKL, TensorFlow, etc. -> evita oversubscription en 2 vCPU
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=2 \
    NUMEXPR_NUM_THREADS=2 \
    TF_NUM_INTRAOP_THREADS=2 \
    TF_NUM_INTEROP_THREADS=1 \
    # Ajustable: workers para uvicorn (por defecto 1, puedes subir a 2 si necesitas más concurrencia)
    UVICORN_WORKERS=1

WORKDIR /app

# --- Dependencias del sistema (mínimas) ---
RUN apt-get update && apt-get install -y --no-install-recommends \
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

# Copiar requirements primero para cache de Docker
COPY requirements.txt /app/requirements.txt

# Actualizar pip y preinstalar numpy compatible con TF (evita recompilaciones)
RUN python -m pip install --upgrade pip setuptools wheel \
 && python -m pip install --no-cache-dir "numpy>=1.26.0,<1.27"

# Instalar dependencias del proyecto
RUN python -m pip install --no-cache-dir -r /app/requirements.txt

# Copiar código (al final para aprovechar cache)
COPY . /app

# Crear directorios de cache para Keras/weights (montables como volúmenes si quieres persistir)
RUN mkdir -p /app/.keras /app/argos_data && chmod -R 755 /app/.keras /app/argos_data

# (Opcional) Intento de pre-descarga seguro: no falla el build si hay problemas de red.
# Ten en cuenta que descargar pesos grandes en build aumenta mucho el tamaño de la imagen.
RUN python - <<'PY'
import os
os.makedirs(os.environ.get("KERAS_HOME", "/app/.keras"), exist_ok=True)
print("KERAS_HOME =", os.environ.get("KERAS_HOME"))
try:
    # Sólo intenta si TensorFlow y Keras ya están instalados en la imagen
    from tensorflow.keras.applications import EfficientNetV2L
    print("-> Intentando descargar EfficientNetV2L weights (ImageNet)...")
    EfficientNetV2L(weights='imagenet', include_top=True, input_shape=(480,480,3))
    print("-> Pesos descargados correctamente (se guardarán en KERAS_HOME).")
except Exception as e:
    print("Warning: no se descargaron pesos en build (esto es intencional para mantener imagen pequeña):", e)

try:
    import nltk
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    print("-> NLTK datasets descargados.")
except Exception as e:
    print("Warning: fallo descargando NLTK durante build:", e)
PY

EXPOSE ${PORT}

# CMD: uvicorn. UVICORN_WORKERS configurable desde Railway: e.g. set UVICORN_WORKERS=2 si quieres 2 procesos.
# Recomendación para 2 vCPU: mantener UVICORN_WORKERS=1 y controlar hilos (cooperativo + TF) o probar 2 si observas que 1 se queda corto.
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT} --loop asyncio --workers ${UVICORN_WORKERS}"]
