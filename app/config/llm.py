# app/config/llm.py

import os
import asyncio
from datetime import datetime
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import Optional, Dict

# Muat variabel lingkungan
load_dotenv()

# Impor LLM dari langchain (pastikan package langchain sudah terinstall)
from langchain_google_genai import ChatGoogleGenerativeAI

# Impor fungsi rotasi API key dan tipe dari modul yang telah dibuat
from app.utils.api_key_rotation import initialize_key_state, get_current_key, handle_error, reset_counters
from app.types.main_types import KeyState

# Definisikan konfigurasi LLM menggunakan Pydantic
class LLMConfig(BaseModel):
    model: str
    temperature: float
    maxRetries: int
    retryDelay: int
    timeout: int

default_config = LLMConfig(
    model="gemini-2.0-flash",
    temperature=0,
    maxRetries=50,
    retryDelay=60000,
    timeout=3000000,
)

# Variabel state global
key_state: Optional[KeyState] = None
is_initialized: bool = False

async def initialize_llm() -> None:
    global key_state, is_initialized
    if is_initialized:
        return
    key_state = await initialize_key_state()
    is_initialized = True

async def get_current_llm(config: Optional[Dict] = None) -> ChatGoogleGenerativeAI:
    global key_state, is_initialized
    if not is_initialized:
        await initialize_llm()
    
    key_response = await get_current_key(key_state)
    if not key_response.success or not key_response.data:
        raise Exception(key_response.error or "Failed to get API key")
    
    api_key, new_state = key_response.data
    key_state = new_state
    
    llm_config_data = default_config.dict()
    if config:
        llm_config_data.update(config)
    llm_config = LLMConfig(**llm_config_data)
    
    return ChatGoogleGenerativeAI(
        model=llm_config.model,
        temperature=llm_config.temperature,
        api_key=api_key
    )

async def handle_llm_error(error: Exception) -> None:
    global key_state, is_initialized
    if not is_initialized:
        raise Exception("LLM system not initialized")
    
    error_response = await handle_error(key_state, error)
    if not error_response.success or not error_response.data:
        raise Exception(error_response.error or "Failed to handle error")
    
    _, new_state = error_response.data
    key_state = new_state

async def reset_llm_state() -> None:
    global key_state, is_initialized
    if not is_initialized:
        raise Exception("LLM system not initialized")
    
    reset_response = await reset_counters(key_state)
    if not reset_response.success or not reset_response.data:
        raise Exception(reset_response.error or "Failed to reset counters")
    
    key_state = reset_response.data

def get_llm_status() -> dict:
    global key_state, is_initialized
    return {
        "initialized": is_initialized,
        "currentConfig": default_config.dict(),
        "keyState": key_state.dict() if (is_initialized and key_state) else None,
    }

async def create_custom_llm(config: dict) -> ChatGoogleGenerativeAI:
    return await get_current_llm(config)

# Pastikan LLM diinisialisasi ketika modul diimpor (jika event loop aktif)
try:
    loop = asyncio.get_running_loop()
except RuntimeError:
    loop = None

if loop:
    asyncio.create_task(initialize_llm())



# import os
# import asyncio
# from datetime import datetime
# from dotenv import load_dotenv
# from pydantic import BaseModel
# from typing import Optional, Any, Dict

# # Muat variabel lingkungan
# load_dotenv()

# # Impor LLM dari langchain (pastikan package langchain sudah terinstall)
# from langchain_google_genai import ChatGoogleGenerativeAI

# # Impor fungsi rotasi API key dan tipe dari modul yang telah dibuat
# from app.utils.api_key_rotation import initialize_key_state, get_current_key, handle_error, reset_counters
# from app.types.main_types import KeyState, ServiceResponse

# # Definisikan konfigurasi LLM menggunakan Pydantic
# class LLMConfig(BaseModel):
#     model: str
#     temperature: float
#     maxRetries: int
#     retryDelay: int
#     timeout: int

# default_config = LLMConfig(
#     model="gemini-1.5-flash",
#     temperature=0,
#     maxRetries=2,
#     retryDelay=1000,
#     timeout=30000,
# )

# # Variabel state global
# key_state: Optional[KeyState] = None
# is_initialized: bool = False

# async def initialize_llm() -> ServiceResponse[None]:
#     global key_state, is_initialized
#     try:
#         if is_initialized:
#             return ServiceResponse(
#                 success=True,
#                 data=None,
#                 metadata={
#                     "processing_time": 0,
#                     "api_key_used": key_state.current_index if key_state else -1,
#                     "timestamp": datetime.utcnow().isoformat()
#                 }
#             )
#         key_state = await initialize_key_state()
#         is_initialized = True
#         return ServiceResponse(
#             success=True,
#             data=None,
#             metadata={
#                 "processing_time": 0,
#                 "api_key_used": key_state.current_index,
#                 "timestamp": datetime.utcnow().isoformat()
#             }
#         )
#     except Exception as e:
#         return ServiceResponse(success=False, error=str(e), metadata={})

# async def get_current_llm(config: Optional[Dict] = None) -> ServiceResponse[ChatGoogleGenerativeAI]:
#     global key_state, is_initialized
#     try:
#         if not is_initialized:
#             await initialize_llm()
#         key_response = await get_current_key(key_state)
#         if not key_response.success or not key_response.data:
#             raise Exception(key_response.error or "Failed to get API key")
#         api_key, new_state = key_response.data
#         key_state = new_state
#         # Gabungkan konfigurasi default dengan override jika ada
#         llm_config_data = default_config.dict()
#         if config:
#             llm_config_data.update(config)
#         llm_config = LLMConfig(**llm_config_data)
#         # Buat instance LLM dengan konfigurasi yang ditentukan
#         llm = ChatGoogleGenerativeAI(
#             model_name=llm_config.model,
#             temperature=llm_config.temperature,
#             max_retries=llm_config.maxRetries,
#             api_key=api_key
#             # Parameter timeout dapat ditambahkan jika didukung oleh class tersebut
#         )
#         return ServiceResponse(
#             success=True,
#             data=llm,
#             metadata={
#                 "processing_time": 0,
#                 "api_key_used": key_state.current_index,
#                 "timestamp": datetime.utcnow().isoformat()
#             }
#         )
#     except Exception as e:
#         return ServiceResponse(success=False, error=str(e), metadata={})

# async def handle_llm_error(error: Any) -> ServiceResponse[None]:
#     global key_state, is_initialized
#     try:
#         if not is_initialized:
#             raise Exception("LLM system not initialized")
#         error_response = await handle_error(key_state, error)
#         if not error_response.success or not error_response.data:
#             raise Exception(error_response.error or "Failed to handle error")
#         _, new_state = error_response.data
#         key_state = new_state
#         return ServiceResponse(
#             success=True,
#             data=None,
#             metadata={
#                 "processing_time": 0,
#                 "api_key_used": key_state.current_index,
#                 "timestamp": datetime.utcnow().isoformat()
#             }
#         )
#     except Exception as e:
#         return ServiceResponse(success=False, error=str(e), metadata={})

# async def reset_llm_state() -> ServiceResponse[None]:
#     global key_state, is_initialized
#     try:
#         if not is_initialized:
#             raise Exception("LLM system not initialized")
#         reset_response = await reset_counters(key_state)
#         if not reset_response.success or not reset_response.data:
#             raise Exception(reset_response.error or "Failed to reset counters")
#         key_state = reset_response.data
#         return ServiceResponse(
#             success=True,
#             data=None,
#             metadata={
#                 "processing_time": 0,
#                 "api_key_used": key_state.current_index,
#                 "timestamp": datetime.utcnow().isoformat()
#             }
#         )
#     except Exception as e:
#         return ServiceResponse(success=False, error=str(e), metadata={})

# def get_llm_status() -> ServiceResponse[dict]:
#     global key_state, is_initialized
#     return ServiceResponse(
#         success=True,
#         data={
#             "initialized": is_initialized,
#             "currentConfig": default_config.dict(),
#             "keyState": key_state.dict() if (is_initialized and key_state) else None,
#         },
#         metadata={
#             "processing_time": 0,
#             "api_key_used": key_state.current_index if (is_initialized and key_state) else -1,
#             "timestamp": datetime.utcnow().isoformat()
#         }
#     )

# async def create_custom_llm(config: dict) -> ServiceResponse[ChatGoogleGenerativeAI]:
#     return await get_current_llm(config)

# # Pastikan LLM diinisialisasi ketika modul diimpor (jika event loop aktif)
# try:
#     loop = asyncio.get_running_loop()
# except RuntimeError:
#     loop = None

# if loop:
#     asyncio.create_task(initialize_llm())
