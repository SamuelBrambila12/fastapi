# Dockerfile mínimo para FastAPI + TensorFlow + Argos Translate (Python 3.11)

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=8000 \
    KERAS_HOME=/app/.keras \
    TF_CPP_MIN_LOG_LEVEL=2 \
    ARGOS_TRANSLATE_PACKAGE_DIR=/app/argos_data

# Instalar SOLO librerías de runtime (no -dev) para reducir tamaño
RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg62-turbo \
    libpng16-16 \
    zlib1g \
    libtiff5 \
    libopenblas0 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copiar requirements y preparar pip
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip setuptools wheel

# Preinstalar numpy compatible con TF (acelera resolución y evita compilación)
RUN pip install --no-cache-dir "numpy>=1.26.0,<1.27"

# Instalar dependencias del proyecto (incluye TensorFlow y Argos Translate)
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copiar el código fuente (main.py en /app)
COPY . /app

# Directorios de caché (montables como volumen si deseas persistir)
RUN mkdir -p /app/.keras /app/argos_data

# NOTA: Evitamos pre-descargar pesos de EfficientNetV2-L y datasets NLTK en build
# para mantener la imagen < 4GB. Se descargarán en el primer arranque si son necesarios.

EXPOSE ${PORT}

# Ejecutar Uvicorn (main.py debe definir app = FastAPI(...))
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT} --loop asyncio --workers 1"]
