from fastapi import FastAPI, HTTPException
from motor.motor_asyncio import AsyncIOMotorClient
from app.api.routes import initialize_rag, router as rag_router
from app.configuration import settings

app = FastAPI()

@app.get("/")
def root():
    return {"message": "API is running"}

# Jalankan initialize_rag saat startup
@app.on_event("startup")
async def startup_event():
    try:
        await initialize_rag()
        print("✅ RAG system initialized successfully!")
    except Exception as e:
        print(f"❌ Failed to initialize RAG system: {str(e)}")

app.include_router(rag_router, prefix="/api/rag")
