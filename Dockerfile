# Dockerfile para FastAPI + TensorFlow (Python 3.11)
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PORT=8000

# Instalar dependencias del sistema necesarias
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    pkg-config \
    libjpeg-dev \
    zlib1g-dev \
    libpng-dev \
    libtiff5-dev \
    libopenblas-dev \
    wget \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copiar requirements y usar pip (copiamos primero para aprovechar cache)
COPY requirements.txt .

RUN pip install --upgrade pip setuptools wheel

# Instala numpy compatible con tensorflow 2.20 antes de instalar el resto
# (evita conflictos de resolución)
RUN pip install --no-cache-dir "numpy>=1.26.0,<1.27"

# Ahora instala el resto de paquetes (requirements.txt ya no contiene numpy)
RUN pip install --no-cache-dir -r requirements.txt

# Copiar todo el código
COPY . .

EXPOSE ${PORT}

# Comando de ejecución (usa $PORT que Railway inyecta)
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT} --loop asyncio --workers 1"]
