# Dockerfile para FastAPI + TensorFlow + Argos Translate (Python 3.11)

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000 \
    KERAS_HOME=/app/.keras \
    TF_CPP_MIN_LOG_LEVEL=2 \
    ARGOS_TRANSLATE_PACKAGE_DIR=/app/argos_data

# Dependencias del sistema necesarias (Pillow, TensorFlow CPU)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    pkg-config \
    libjpeg-dev \
    zlib1g-dev \
    libpng-dev \
    libtiff5-dev \
    libopenblas-dev \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copiar requirements para aprovechar cache de Docker
COPY requirements.txt /app/requirements.txt

RUN pip install --upgrade pip setuptools wheel

# Forzar una versión de numpy compatible antes de instalar TF (evita conflictos)
RUN pip install --no-cache-dir "numpy>=1.26.0,<1.27"

# Instalar dependencias del proyecto (requirements.txt NO debe fijar numpy si haces lo anterior)
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copiar el código fuente (main.py y utilidades deben estar en la raíz del contexto)
COPY . /app

# Crear directorios de caché (Keras y Argos)
RUN mkdir -p /app/.keras /app/argos_data

# PRE-DOWNLOAD opcional: descargar pesos del modelo y datasets de NLTK durante el build.
# Para minimizar tamaño de la imagen, puedes comentar estas dos líneas y permitir descarga en el primer arranque.
RUN python -c "import os; os.makedirs(os.environ.get('KERAS_HOME','/app/.keras'), exist_ok=True); print('KERAS_HOME =', os.environ.get('KERAS_HOME')); import tensorflow as tf; from tensorflow.keras.applications import EfficientNetV2L; print('-> Descargando EfficientNetV2L weights (ImageNet)...'); EfficientNetV2L(weights='imagenet', include_top=True, input_shape=(480,480,3)); print('-> Pesos descargados.')" || true
RUN python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4'); print('-> NLTK datasets descargados.')" || true

EXPOSE ${PORT}

# Ejecutar uvicorn (usa la variable PORT inyectada por Railway al deploy)
# Asegúrate de que main.py esté en /app y defina `app = FastAPI(...)`
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT} --loop asyncio --workers 1"]
