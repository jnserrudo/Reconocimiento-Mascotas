# main.py (Versión 6.1 - Integración + Endpoint de Pruebas)

import os
import io
import json
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Literal, Optional
from dotenv import load_dotenv
from supabase import create_client, Client
from uuid import UUID # Importante para validar el ID del reporte

import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing import image as keras_image

# --- Cargar Configuración ---
load_dotenv()
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
base_model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')

print("✅ Servicio de IA iniciado y conectado.")

# --- Configuración de la App ---
app = FastAPI(
    title="Servicio de IA para Reconocimiento de Mascotas",
    description="Provee endpoints para la integración (calcular vector, buscar coincidencias) y un endpoint de prueba para publicar reportes completos."
)

origins = [
    "http://localhost:3000", # Para tu desarrollo local del front
    "http://localhost:5174", # Otro puerto común de Vite
    "https://reconocimiento-mascotas.onrender.com" # La URL de tu front cuando lo despliegues
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Modelos de Datos ---
class VectorResponse(BaseModel):
    vector: List[float]

class MatchResult(BaseModel):
    id: str
    title: Optional[str] = None
    description: Optional[str] = None
    image_url: str
    similarity: float

class PublishResult(BaseModel):
    status: str
    report_id: str
    image_url: str



# NUEVO MODELO para la respuesta del detalle del reporte
class ReportImageDetail(BaseModel):
    id: UUID
    image_url: str
    is_primary: bool

class ReportDetail(BaseModel):
    id: UUID
    title: str
    description: str
    species: str
    location: str
    report_type: str
    status: str
    contact_info: dict
    report_images: List[ReportImageDetail] # Anidamos las imágenes


# --- Lógica Central ---
def extract_features(img_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB').resize((224, 224))
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = base_model.predict(img_array, verbose=0)
    return features.flatten()




@app.get("/report/{report_id}", response_model=ReportDetail, tags=["Integración - Datos"])
async def get_report_details(report_id: UUID):
    """
    Obtiene todos los detalles de un reporte específico, incluyendo todas sus imágenes.
    """
    try:
        # Hacemos una consulta a la tabla 'reports' y le pedimos que incluya
        # todos los datos relacionados de la tabla 'report_images'
        response = supabase.table("reports").select(
            "*, report_images(*)"
        ).eq("id", str(report_id)).single().execute()

        if not response.data:
            raise HTTPException(status_code=404, detail="Reporte no encontrado")

        return response.data
    except Exception as e:
        # Captura errores genéricos o si el UUID es inválido
        raise HTTPException(status_code=500, detail=f"Error al obtener el reporte: {e}")
        
# =================================================================
# ENDPOINTS OFICIALES PARA LA INTEGRACIÓN CON EL EQUIPO
# =================================================================

@app.post("/calculate_vector", response_model=VectorResponse, tags=["Integración - Backend Java"])
async def calculate_vector(file: UploadFile = File(...)):
    """
    Recibe una imagen y devuelve su vector numérico. Para ser usado por el backend principal.
    """
    image_bytes = await file.read()
    vector = extract_features(image_bytes)
    return {"vector": vector.tolist()}

@app.post("/find_proactive_matches", response_model=List[MatchResult], tags=["Integración - Frontend"])
async def find_proactive_matches(
    search_in_type: Literal['perdido', 'encontrado'] = Form(...),
    file: UploadFile = File(...)
):
    """
    Recibe una imagen y busca coincidencias en la DB. Para ser usado por el frontend.
    """
    image_bytes = await file.read()
    query_vector = extract_features(image_bytes)
    try:
        response = supabase.rpc('match_reports_advanced', {
            'query_vector': query_vector.tolist(),
            'match_threshold': 0.60,
            'report_type_to_match': search_in_type,
            'result_limit': 5
        }).execute()
        return response.data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en la búsqueda RPC: {e}")

# =================================================================
# ENDPOINT DE PRUEBA PARA TU USO PERSONAL
# =================================================================

@app.post("/publish_report_test", response_model=PublishResult, tags=["Pruebas - IA Dev"])
async def publish_report_test(
    report_type: Literal['perdido', 'encontrado'] = Form(...),
    species: Literal['perro', 'gato', 'otro'] = Form(...),
    title: str = Form(...),
    description: str = Form(...),
    location: str = Form(...),
    contact_info: str = Form(...),
    file: UploadFile = File(...)
):
    """
    (SOLO PARA PRUEBAS) Endpoint completo para publicar un aviso.
    Permite probar la lógica de subida y guardado de forma aislada.
    """
    print("\n--- EJECUTANDO ENDPOINT DE PRUEBA: /publish_report_test ---")
    
    # Paso 1: Insertar datos en la tabla 'reports'
    try:
        contact_info_json = json.loads(contact_info)
        report_data = {
            "report_type": report_type, "species": species, "title": title,
            "description": description, "location": location, "contact_info": contact_info_json
        }
        response = supabase.table("reports").insert(report_data).execute()
        if not response.data:
            raise HTTPException(status_code=500, detail="Fallo al crear el reporte en DB.")
        report_id = response.data[0]['id']
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al insertar en 'reports': {e}")

    # Paso 2 y 3: Subir imagen y calcular vector
    image_bytes = await file.read()
    vector = extract_features(image_bytes)
    file_path = f"{report_type}/{report_id}/{file.filename}"
    
    try:
        supabase.storage.from_("images").upload(file_path, image_bytes, {"content-type": file.content_type})
        image_url = supabase.storage.from_("images").get_public_url(file_path)
    except Exception as e:
        supabase.table("reports").delete().eq("id", report_id).execute()
        raise HTTPException(status_code=500, detail=f"Error al subir imagen a Storage: {e}")

    # Paso 4: Insertar datos de la imagen en 'report_images'
    try:
        image_data = {
            "report_id": report_id, "image_url": image_url,
            "is_primary": True, "vector": vector.tolist()
        }
        image_response = supabase.table("report_images").insert(image_data).execute()
        if not image_response.data:
            raise HTTPException(status_code=500, detail="Fallo al guardar la imagen del reporte.")
    except Exception as e:
        supabase.table("reports").delete().eq("id", report_id).execute()
        supabase.storage.from_("images").remove([file_path])
        raise HTTPException(status_code=500, detail=f"Error al insertar en 'report_images': {e}")

    print("--- FIN DE EJECUCIÓN DEL ENDPOINT DE PRUEBA ---")
    return {"status": "ok", "report_id": report_id, "image_url": image_url}


