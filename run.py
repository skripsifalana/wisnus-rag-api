#!/usr/bin/env python3
"""
Simple script to run the Wisnus RAG API application
"""

import uvicorn
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

if __name__ == "__main__":
    # Get configuration from environment variables
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    reload = os.getenv("RELOAD", "true").lower() == "true"
    
    print(f"🚀 Starting Wisnus RAG API...")
    print(f"📍 Host: {host}")
    print(f"🔌 Port: {port}")
    print(f"🔄 Auto-reload: {reload}")
    print(f"📖 API Documentation: http://{host}:{port}/docs")
    print(f"🔧 Alternative docs: http://{host}:{port}/redoc")
    print("-" * 50)
    
    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    ) 