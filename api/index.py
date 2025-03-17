from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import os
import sys

# Add root directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import main application
from app.main import app as app_main

# Create new FastAPI instance for Vercel
app = FastAPI()

# Get CORS configuration from environment variable
# Default to localhost:3000 if not provided
cors_origins_str = os.getenv("CORS_ORIGINS", "http://localhost:3000")
cors_origins = [origin.strip() for origin in cors_origins_str.split(",")]

# Add CORS middleware with specific configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

# Root endpoint
@app.get("/")
async def root():
    return {"message": "API is running"}

# Delegate requests to main application
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    if request.url.path != "/":
        return await app_main(request.scope, request.receive, request.send)
    response = await call_next(request)
    return response