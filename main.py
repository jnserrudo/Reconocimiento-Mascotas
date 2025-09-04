# main.py (Versión 5.0 - Integrado con Esquema Completo de Supabase)

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
from pgvector.sqlalchemy import Vector

import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing import image as keras_image

# --- Cargar Variables de Entorno y Conectar a Supabase ---
load_dotenv()
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY") # Asegúrate de que esta sea tu service_role key

if not SUPABASE_URL or not SUPABASE_KEY:
    raise Exception("Error: Las credenciales de Supabase no están configuradas en el archivo .env")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
print("✅ Conectado a Supabase.")

# --- Configuración de la App FastAPI ---
app = FastAPI(
    title="API de Mascotas v5.0",
    description="Backend completo con base de datos relacional en Supabase y búsqueda vectorial."
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Para desarrollo. En producción, especifica tus dominios.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Cargar Modelo de IA ---
base_model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')
print("✅ Modelo MobileNetV2 cargado.")

def extract_features(img_bytes: bytes) -> np.ndarray:
    """Calcula el vector de características de una imagen."""
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB').resize((224, 224))
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = base_model.predict(img_array, verbose=0)
    return features.flatten()

# --- Modelos de Datos Pydantic para la API ---
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

# --- Endpoints de la API ---

@app.post("/publish_report", response_model=PublishResult, status_code=201)
async def publish_report(
    report_type: Literal['perdido', 'encontrado'] = Form(...),
    species: Literal['perro', 'gato', 'otro'] = Form(...),
    title: str = Form(...),
    description: str = Form(...),
    location: str = Form(...),
    contact_info: str = Form(...),
    file: UploadFile = File(...)
):
    print("\n--- INICIANDO /publish_report (v5.1 CORREGIDO) ---")
    
    # Paso 1: Insertar datos en la tabla 'reports'
    try:
        print("Paso 1: Intentando insertar en la tabla 'reports'...")
        contact_info_json = json.loads(contact_info)
        report_data = {
            "report_type": report_type, "species": species, "title": title,
            "description": description, "location": location, "contact_info": contact_info_json
        }
        print(f"  > Datos a insertar: {report_data}")
        
        # --- CORRECCIÓN CLAVE AQUÍ ---
        # Ejecutamos el insert directamente, sin encadenar .select()
        response = supabase.table("reports").insert(report_data).execute()
        
        # La librería v2 devuelve los datos insertados en response.data
        if not response.data:
            print("❌ ERROR: La inserción en 'reports' no devolvió datos.")
            raise HTTPException(status_code=500, detail="Fallo al crear el reporte en DB (paso 1).")
        
        report_id = response.data[0]['id']
        print(f"✅ Paso 1 completado. Reporte creado con ID: {report_id}")

    except json.JSONDecodeError:
        print("❌ ERROR: El formato de contact_info no es un JSON válido.")
        raise HTTPException(status_code=400, detail="JSON de contacto mal formado.")
    except Exception as e:
        print(f"❌ ERROR CRÍTICO en el Paso 1 (insertar en reports): {e}")
        raise HTTPException(status_code=500, detail=str(e))

    # El resto de la función es igual y debería funcionar ahora
    print("Paso 2: Leyendo imagen y calculando vector...")
    image_bytes = await file.read()
    vector = extract_features(image_bytes)
    file_path = f"{report_type}/{report_id}/{file.filename}"
    print(f"  > Vector calculado. Ruta de archivo: {file_path}")
    
    try:
        print("Paso 3: Intentando subir imagen a Supabase Storage...")
        supabase.storage.from_("images").upload(file_path, image_bytes, {"content-type": file.content_type})
        image_url = supabase.storage.from_("images").get_public_url(file_path)
        print(f"✅ Paso 3 completado. Imagen subida a: {image_url}")
    except Exception as e:
        print(f"❌ ERROR CRÍTICO en el Paso 3 (subir a Storage): {e}")
        supabase.table("reports").delete().eq("id", report_id).execute()
        raise HTTPException(status_code=500, detail=str(e))

    try:
        print("Paso 4: Intentando insertar en la tabla 'report_images'...")
        image_data = {
            "report_id": report_id, "image_url": image_url,
            "is_primary": True, "vector": vector.tolist()
        }
        print(f"  > Datos de imagen a insertar: {{report_id: {report_id}, ...}}")
        image_response = supabase.table("report_images").insert(image_data).execute()
        
        if not image_response.data:
            print("❌ ERROR: La inserción en 'report_images' no devolvió datos.")
            raise HTTPException(status_code=500, detail="Fallo al guardar la imagen del reporte (paso 4).")
        print("✅ Paso 4 completado. Imagen y vector guardados.")

    except Exception as e:
        print(f"❌ ERROR CRÍTICO en el Paso 4 (insertar en report_images): {e}")
        supabase.table("reports").delete().eq("id", report_id).execute()
        supabase.storage.from_("images").remove([file_path])
        raise HTTPException(status_code=500, detail=str(e))

    print("--- /publish_report COMPLETADO CON ÉXITO ---")
    return {"status": "ok", "report_id": report_id, "image_url": image_url}


@app.post("/find_proactive_matches", response_model=List[MatchResult])
async def find_proactive_matches(
    search_in_type: Literal['perdido', 'encontrado'] = Form(...),
    file: UploadFile = File(...)
):
    """
    Busca coincidencias llamando a la función RPC de la base de datos.
    Esto es extremadamente eficiente, ya que todo el cálculo de similitud se hace en PostgreSQL.
    """
    image_bytes = await file.read()
    query_vector = extract_features(image_bytes)

    try:
        response = supabase.rpc('match_reports_advanced', {
            'query_vector': query_vector.tolist(),
            'match_threshold': 0.60, # Umbral de similitud
            'report_type_to_match': search_in_type,
            'result_limit': 5 # Devolver como máximo 5 coincidencias
        }).execute()
        
        print(f"🔍 Búsqueda RPC ejecutada. Encontradas {len(response.data)} coincidencias.")
        return response.data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en la búsqueda por similitud RPC: {e}")