from fastapi import FastAPI, HTTPException
from motor.motor_asyncio import AsyncIOMotorClient
from fastapi.middleware.cors import CORSMiddleware
import os
from app.api.routes import initialize_rag, router as rag_router
from app.configuration import settings

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

@app.get("/")
def root():
    return {"message": "API is running"}

# Initialize RAG system on startup
@app.on_event("startup")
async def startup_event():
    try:
        await initialize_rag()
        print("✅ RAG system initialized successfully!")
    except Exception as e:
        print(f"❌ Failed to initialize RAG system: {str(e)}")

app.include_router(rag_router, prefix="/api/rag")