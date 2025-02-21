import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    MONGO_URI: str = os.getenv("MONGO_URI", "mongodb://localhost:27017")
    MONGO_DB_NAME: str = os.getenv("MONGO_DB_NAME", "rag_db")
    LANGSMITH_TRACING: bool = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"
    # LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "openai")
    # VECTOR_STORE_URI: str = os.getenv("VECTOR_STORE_URI", "mongodb://localhost:27017/vectorstore")

settings = Settings()