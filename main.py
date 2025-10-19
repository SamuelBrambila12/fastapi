from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import json
import os
from typing import List, Dict, Optional
import logging
import asyncio
from pydantic import BaseModel
from utils.example_generator import generate_examples


# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Variables globales
model = None
MODEL_INPUT_SIZE = (480, 480)
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

class ModelManager:
    def __init__(self):
        self.model = None
        self.model_loaded = False
        self.loading_lock = asyncio.Lock()
    
    async def load_model(self):
        """Carga el modelo preentrenado de forma asíncrona"""
        async with self.loading_lock:
            if self.model_loaded:
                return True
                
            try:
                logger.info("Cargando modelo EfficientNetV2-L (ImageNet)...")
                # Ejecutar la carga del modelo en un thread para no bloquear el event loop
                loop = asyncio.get_event_loop()
                self.model = await loop.run_in_executor(
                    None,
                    lambda: tf.keras.applications.EfficientNetV2L(
                        weights='imagenet',
                        include_top=True,
                        input_shape=(480, 480, 3)
                    )
                )
                self.model_loaded = True
                logger.info("Modelo cargado exitosamente")
                return True
            except Exception as e:
                logger.error(f"Error cargando modelo: {e}")
                self.model_loaded = False
                return False
    
    async def get_model(self):
        """Obtiene el modelo, cargándolo si es necesario"""
        if not self.model_loaded:
            await self.load_model()
        return self.model

# Instancia global del manejador de modelo
model_manager = ModelManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manejador del ciclo de vida de la aplicación"""
    # Startup
    logger.info("Iniciando aplicación...")
    await model_manager.load_model()
    yield
    # Shutdown
    logger.info("Cerrando aplicación...")

app = FastAPI(
    title="LexIA API",
    description="API para clasificación de imágenes usando TensorFlow",
    version="1.0.0",
    lifespan=lifespan
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especifica dominios específicos
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

def validate_image_file(file: UploadFile) -> bool:
    """Valida si el archivo es una imagen válida con debug mejorado"""
    logger.info(f"🔍 Validando archivo: {file.filename}")
    logger.info(f"📋 Content-Type: {file.content_type}")
    
    # Lista de content-types permitidos (más completa)
    allowed_content_types = [
        "image/jpeg", "image/jpg", "image/png", "image/bmp", 
        "image/gif", "image/tiff", "image/webp", "image/x-ms-bmp",
        "application/octet-stream"  # A veces los archivos vienen así
    ]
    
    # Lista de extensiones permitidas
    allowed_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp']
    
    # Obtener extensión del archivo
    file_extension = ""
    if file.filename:
        file_extension = os.path.splitext(file.filename.lower())[1]
        logger.info(f"📎 Extensión detectada: {file_extension}")
    
    # Validación más flexible: si tiene extensión válida O content-type válido
    content_type_valid = file.content_type and (
        file.content_type in allowed_content_types or 
        file.content_type.startswith("image/")
    )
    
    extension_valid = file_extension in allowed_extensions
    
    logger.info(f"✅ Content-type válido: {content_type_valid}")
    logger.info(f"✅ Extensión válida: {extension_valid}")
    
    # Aceptar si cualquiera de las dos validaciones pasa
    is_valid = content_type_valid or extension_valid
    
    if not is_valid:
        logger.warning(f"❌ Archivo rechazado:")
        logger.warning(f"   - Content-Type: {file.content_type}")
        logger.warning(f"   - Extensión: {file_extension}")
        logger.warning(f"   - Content-types permitidos: {allowed_content_types}")
        logger.warning(f"   - Extensiones permitidas: {allowed_extensions}")
    
    return is_valid

async def preprocess_image(image: Image.Image) -> np.ndarray:
    """Preprocesa la imagen para el modelo de forma asíncrona"""
    try:
        # Ejecutar el preprocesamiento en un thread para no bloquear
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _preprocess_image_sync, image)
    except Exception as e:
        logger.error(f"Error en preprocesamiento: {e}")
        raise HTTPException(status_code=400, detail=f"Error procesando imagen: {str(e)}")

def _preprocess_image_sync(image: Image.Image) -> np.ndarray:
    """Función síncrona para el preprocesamiento"""
    # Convertir a RGB si es necesario
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Redimensionar a 480x480
    image = image.resize(MODEL_INPUT_SIZE, Image.Resampling.LANCZOS)
    
    # Convertir a array numpy
    img_array = tf.keras.utils.img_to_array(image)
    
    # Expandir dimensiones para batch
    img_array = tf.expand_dims(img_array, 0)
    
    # Preprocesar según EfficientNetV2
    img_array = tf.keras.applications.efficientnet_v2.preprocess_input(img_array)
    
    return img_array

async def decode_predictions(predictions: np.ndarray, top: int = 5) -> List[Dict]:
    """Decodifica las predicciones del modelo de forma asíncrona"""
    try:
        loop = asyncio.get_event_loop()
        decoded = await loop.run_in_executor(
            None, 
            lambda: tf.keras.applications.imagenet_utils.decode_predictions(predictions, top=top)
        )
        
        results = []
        for pred in decoded[0]:
            results.append({
                "class_id": pred[0],
                "class_name": pred[1].replace('_', ' ').title(),
                "confidence": float(pred[2]),
                "confidence_percent": f"{float(pred[2]) * 100:.2f}%"
            })
        
        return results
    except Exception as e:
        logger.error(f"Error decodificando predicciones: {e}")
        raise HTTPException(status_code=500, detail=f"Error procesando predicciones: {str(e)}")

@app.get("/")
async def root():
    """Endpoint raíz"""
    return {
        "message": "LexIA (VC) API está funcionando",
        "status": "ok",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "predict_batch": "/predict/batch",
            "examples": "/examples",
            "docs": "/docs",
            "redoc": "/redoc"
        }
    }

@app.get("/health")
async def health_check():
    """Verificar el estado de la API y el modelo"""
    return {
        "status": "healthy" if model_manager.model_loaded else "unhealthy",
        "model_loaded": model_manager.model_loaded,
        "model_type": "EfficientNetV2L",
        "input_size": MODEL_INPUT_SIZE,
        "tensorflow_version": tf.__version__,
        "max_file_size_mb": MAX_FILE_SIZE // (1024 * 1024)
    }

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    """Endpoint para clasificar una imagen"""
    
    # Verificar que el modelo esté cargado
    model = await model_manager.get_model()
    if model is None:
        raise HTTPException(status_code=500, detail="Modelo no disponible")
    
    # Validar archivo
    if not validate_image_file(file):
        raise HTTPException(
            status_code=400, 
            detail="Archivo inválido. Solo se permiten imágenes (jpg, png, bmp, gif, tiff)"
        )
    
    # Verificar tamaño del archivo
    contents = await file.read()
    file_size = len(contents)
    
    if file_size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400, 
            detail=f"Archivo muy grande. Máximo permitido: {MAX_FILE_SIZE // (1024*1024)}MB"
        )
    
    if file_size == 0:
        raise HTTPException(status_code=400, detail="Archivo vacío")
    
    try:
        # Cargar y procesar imagen
        image = Image.open(io.BytesIO(contents))
        processed_image = await preprocess_image(image)
        
        # Realizar predicción de forma asíncrona
        logger.info(f"Procesando imagen: {file.filename}")
        loop = asyncio.get_event_loop()
        predictions = await loop.run_in_executor(
            None, 
            lambda: model.predict(processed_image, verbose=0)
        )
        
        # Decodificar resultados
        results = await decode_predictions(predictions, top=5)
        
        return {
            "success": True,
            "filename": file.filename,
            "file_size_kb": round(file_size / 1024, 2),
            "image_dimensions": f"{image.size[0]}x{image.size[1]}",
            "processed_shape": list(processed_image.shape),
            "predictions": results,
            "top_prediction": {
                "class": results[0]["class_name"],
                "confidence": results[0]["confidence_percent"]
            } if results else None
        }
        
    except Exception as e:
        logger.error(f"Error procesando {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Error procesando imagen: {str(e)}")

class ExamplesRequest(BaseModel):
    word: str
    count: Optional[int] = 6

@app.post("/examples")
async def examples_endpoint(body: ExamplesRequest):
    w = (body.word or '').strip()
    if not w:
        raise HTTPException(status_code=400, detail="'word' es requerido")
    cnt = body.count or 6
    if cnt < 3:
        cnt = 3
    if cnt > 12:
        cnt = 12
    try:
        loop = asyncio.get_event_loop()
        examples = await loop.run_in_executor(None, lambda: generate_examples(w, cnt))
        if not examples:
            raise RuntimeError("No se pudieron generar ejemplos")
        return {"success": True, "word": w, "examples": examples}
    except Exception as e:
        logger.error(f"Error generando ejemplos para '{w}': {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch")
async def predict_batch(files: List[UploadFile] = File(...)):
    """Endpoint para clasificar múltiples imágenes"""
    
    # Verificar que el modelo esté cargado
    model = await model_manager.get_model()
    if model is None:
        raise HTTPException(status_code=500, detail="Modelo no disponible")
    
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Máximo 10 imágenes por batch")
    
    if len(files) == 0:
        raise HTTPException(status_code=400, detail="No se enviaron archivos")
    
    # Procesar imágenes de forma concurrente
    tasks = []
    for i, file in enumerate(files):
        task = process_single_image(model, file, i + 1, len(files))
        tasks.append(task)
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Convertir excepciones a resultados de error
    processed_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            processed_results.append({
                "filename": files[i].filename,
                "success": False,
                "error": str(result),
                "predictions": []
            })
        else:
            processed_results.append(result)
    
    successful_predictions = sum(1 for r in processed_results if r["success"]) 
    
    return {
        "success": True,
        "batch_size": len(files),
        "successful_predictions": successful_predictions,
        "failed_predictions": len(files) - successful_predictions,
        "results": processed_results
    }


async def process_single_image(model, file: UploadFile, current: int, total: int):
    """Procesa una sola imagen para el batch"""
    try:
        logger.info(f"Procesando imagen {current}/{total}: {file.filename}")
        
        if not validate_image_file(file):
            return {
                "filename": file.filename,
                "success": False,
                "error": "Formato de archivo no válido",
                "predictions": []
            }
        
        contents = await file.read()
        if len(contents) > MAX_FILE_SIZE:
            return {
                "filename": file.filename,
                "success": False,
                "error": "Archivo muy grande",
                "predictions": []
            }
        
        image = Image.open(io.BytesIO(contents))
        processed_image = await preprocess_image(image)
        
        # Realizar predicción
        loop = asyncio.get_event_loop()
        predictions = await loop.run_in_executor(
            None, 
            lambda: model.predict(processed_image, verbose=0)
        )
        
        decoded_results = await decode_predictions(predictions, top=3)
        
        return {
            "filename": file.filename,
            "success": True,
            "predictions": decoded_results,
            "top_prediction": decoded_results[0]["class_name"] if decoded_results else "Unknown"
        }
        
    except Exception as e:
        logger.error(f"Error procesando {file.filename}: {e}")
        return {
            "filename": file.filename,
            "success": False,
            "error": str(e),
            "predictions": []
        }

class PronunciationRequest(BaseModel):
    target: str
    recognized: str


def _phonemes_for_text(text: str):
    t = (text or '').strip()
    if not t:
        return []
    # Intentar g2p_en para fonemas
    try:
        from g2p_en import G2p  # type: ignore
        g2p = G2p()
        phones = g2p(t)
        # Limpiar separadores y solo dejar tokens fonéticos
        phones = [p for p in phones if p and p.strip() and not p.isspace()]
        # Normalizar: quitar marcadores de estrés (1,2)
        norm = []
        for p in phones:
            p = p.strip()
            if p in {" ", "  "}:
                continue
            p = ''.join(ch for ch in p if not ch.isdigit())
            if p:
                norm.append(p)
        return norm
    except Exception:
        # Fallback muy básico: usar caracteres como "fonemas"
        return list(t.lower())


def _align(a, b):
    # Alineación por DP (Levenshtein con backtrace)
    n, m = len(a), len(b)
    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1):
        dp[i][0] = i
    for j in range(m+1):
        dp[0][j] = j
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = 0 if a[i-1] == b[j-1] else 1
            dp[i][j] = min(
                dp[i-1][j] + 1,      # delete
                dp[i][j-1] + 1,      # insert
                dp[i-1][j-1] + cost  # substitute
            )
    # backtrace
    i, j = n, m
    pairs = []
    while i > 0 or j > 0:
        if i > 0 and dp[i][j] == dp[i-1][j] + 1:
            pairs.append((a[i-1], None, False))
            i -= 1
        elif j > 0 and dp[i][j] == dp[i][j-1] + 1:
            pairs.append((None, b[j-1], False))
            j -= 1
        else:
            match = (a[i-1] == b[j-1])
            pairs.append((a[i-1], b[j-1], match))
            i -= 1
            j -= 1
    pairs.reverse()
    distance = dp[n][m]
    norm = distance / max(1, max(n, m))
    score = max(0.0, 1.0 - norm)
    return pairs, score, distance


@app.post("/pronunciation/evaluate")
async def pronunciation_evaluate(body: PronunciationRequest):
    target = (body.target or '').strip()
    recognized = (body.recognized or '').strip()
    if not target:
        raise HTTPException(status_code=400, detail="Campo 'target' es requerido")
    try:
        t_ph = _phonemes_for_text(target)
        # Aceptar evaluación incluso si recognized está vacío: devolver feedback completo
        if not recognized:
            feedback = []
            for i, p in enumerate(t_ph):
                feedback.append({
                    "target_phoneme": p,
                    "heard_phoneme": None,
                    "index": i,
                    "hint": "Falta este sonido; asegúrate de pronunciarlo",
                })
            return {
                "success": True,
                "target": target,
                "recognized": recognized,
                "target_phonemes": t_ph,
                "recognized_phonemes": [],
                "score": 0.0,
                "edit_distance": len(t_ph),
                "feedback": feedback,
            }
        r_ph = _phonemes_for_text(recognized)
        pairs, score, distance = _align(t_ph, r_ph)
        feedback = []
        idx = 0
        for a_tok, b_tok, match in pairs:
            if not match:
                fb = {
                    "target_phoneme": a_tok,
                    "heard_phoneme": b_tok,
                    "index": idx,
                    "hint": None,
                }
                # pistas simples
                if a_tok is None:
                    fb["hint"] = "Sonido extra; intenta omitirlo"
                elif b_tok is None:
                    fb["hint"] = "Falta este sonido; asegúrate de pronunciarlo"
                else:
                    fb["hint"] = "Ajusta este sonido para acercarlo al objetivo"
                feedback.append(fb)
            idx += 1
        # Si el puntaje es 0, garantizar retroalimentación completa por cada fonema objetivo
        if score <= 0.0:
            covered = set()
            for fb in feedback:
                try:
                    covered.add(int(fb.get("index", -1)))
                except Exception:
                    pass
            for i, p in enumerate(t_ph):
                if i not in covered:
                    feedback.append({
                        "target_phoneme": p,
                        "heard_phoneme": None,
                        "index": i,
                        "hint": "Falta este sonido; asegúrate de pronunciarlo",
                    })
        return {
            "success": True,
            "target": target,
            "recognized": recognized,
            "target_phonemes": t_ph,
            "recognized_phonemes": r_ph,
            "score": round(score, 3),
            "edit_distance": distance,
            "feedback": feedback,
        }
    except Exception as e:
        logger.error(f"Error evaluando pronunciación: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Manejador global de excepciones"""
    logger.error(f"Error no manejado: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Error interno del servidor",
            "detail": str(exc)
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",  # Usar import string para enable reload
        host="0.0.0.0", 
        port=8000,
        log_level="info",
        reload=True
    )
