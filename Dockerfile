# Dockerfile para FastAPI + TensorFlow (Python 3.11)
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PORT=8000
ENV KERAS_HOME=/app/.keras
ENV TF_CPP_MIN_LOG_LEVEL=2
ENV CUDA_VISIBLE_DEVICES=-1
ENV TF_NUM_INTRAOP_THREADS=2
ENV TF_NUM_INTEROP_THREADS=2
ENV OMP_NUM_THREADS=2

# Dependencias del sistema necesarias (runtime mínimas)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libstdc++6 \
    libgomp1 \
    libjpeg62-turbo \
    zlib1g \
    libpng16-16 \
    libtiff5 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copiar requirements para aprovechar cache de Docker
COPY requirements.txt .

RUN pip install --upgrade pip setuptools wheel

# Forzar una versión de numpy compatible antes de instalar TF (evita conflictos)
RUN pip install --no-cache-dir "numpy>=1.26.0,<1.27"

# Instalar dependencias del proyecto (requirements.txt no debe contener numpy si haces lo anterior)
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código fuente
COPY . .

# Crear directorio de caché de Keras en build (evita problemas de permisos)
RUN mkdir -p /app/.keras

# Nota: Evitamos pre-descargar pesos/modelos/datasets en build para reducir RAM/tiempo; 
# se descargarán en runtime bajo KERAS_HOME cuando el código los necesite.

EXPOSE ${PORT}

# Ejecutar uvicorn (usa la variable PORT inyectada por Railway al deploy)
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT} --loop asyncio --workers 1"]
