from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import asyncio
import os

# Argos Translate
from argostranslate import package, translate

# Directorio para cachear paquetes de Argos (montar como volumen en Docker para persistir)
ARGOS_DATA_DIR = os.environ.get("ARGOS_DATA_DIR", "/app/argos_data")
os.environ.setdefault("ARGOS_TRANSLATE_PACKAGE_DIR", ARGOS_DATA_DIR)

app = FastAPI(
    title="LexIA Translator",
    description="Servicio de traducción EN->ES ligero con Argos Translate",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

class TranslateRequest(BaseModel):
    texts: List[str]

class TranslatorState:
    def __init__(self):
        self._ready = False
        self._lock = asyncio.Lock()

    async def ensure_ready(self):
        async with self._lock:
            if self._ready:
                return
            # Asegurar directorio
            os.makedirs(ARGOS_DATA_DIR, exist_ok=True)
            # Actualizar índice de paquetes
            package.update_package_index()
            # Ver si ya hay instalado EN->ES
            installed = translate.get_installed_languages()
            have_en = any(l.code == "en" for l in installed)
            have_es = any(l.code == "es" for l in installed)

            if not (have_en and have_es):
                # Buscar el/los paquetes EN->ES disponibles en el índice
                available = package.get_available_packages()
                candidates = [p for p in available if p.from_code == "en" and p.to_code == "es"]
                if not candidates:
                    raise RuntimeError("No hay paquetes EN->ES disponibles para Argos Translate")
                # Elegir el paquete con mayor tamaño (por lo general mejor calidad)
                candidates.sort(key=lambda p: getattr(p, 'size', 0), reverse=True)
                pkg = candidates[0]
                # Descargar e instalar
                pkg_path = pkg.download()
                package.install_from_path(pkg_path)

            # Verificar idiomas
            installed = translate.get_installed_languages()
            en_lang = next((l for l in installed if l.code == "en"), None)
            es_lang = next((l for l in installed if l.code == "es"), None)
            if not en_lang or not es_lang:
                raise RuntimeError("Idiomas EN/ES no disponibles tras la instalación del paquete")

            self._ready = True

    def translate_batch(self, texts: List[str]) -> List[str]:
        # Obtener traductor EN->ES
        installed = translate.get_installed_languages()
        en_lang = next(l for l in installed if l.code == "en")
        es_lang = next(l for l in installed if l.code == "es")
        translator = en_lang.get_translation(es_lang)
        # Traducir cada texto
        out = []
        for t in texts:
            s = (t or '').strip()
            if not s:
                out.append(s)
                continue
            out.append(translator.translate(s))
        return out

state = TranslatorState()

@app.get("/health")
async def health():
    return {"status": "ok", "ready": state._ready}

@app.post("/translate")
async def translate_endpoint(body: TranslateRequest):
    if not body.texts:
        raise HTTPException(status_code=400, detail="'texts' no puede estar vacío")
    try:
        await state.ensure_ready()
        # Ejecutar en threadpool por seguridad (opcional)
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, lambda: state.translate_batch(body.texts))
        return {"success": True, "translations": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
