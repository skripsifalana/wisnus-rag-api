from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import os
import sys

# Tambahkan direktori root ke sys.path agar bisa import modul
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import aplikasi utama
from app.main import app as app_main

# Buat instance FastAPI baru untuk Vercel
app = FastAPI()

# Tambahkan middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Root endpoint untuk health check
@app.get("/")
async def root():
    return {"message": "API is running"}

# Delegasikan semua route ke aplikasi utama
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    if request.url.path != "/":
        return await app_main(request.scope, request.receive, request.send)
    response = await call_next(request)
    return response