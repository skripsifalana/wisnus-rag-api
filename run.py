# run.py

#!/usr/bin/env python3
"""
Script untuk menjalankan server RAG API
"""

import os
import sys
import uvicorn
from pathlib import Path

def setup_environment():
    """Setup environment variables dan working directory"""
    # Set working directory ke root project
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    # Set default environment variables jika belum ada
    if not os.getenv("PORT"):
        os.environ["PORT"] = "8000"
    
    if not os.getenv("HOST"):
        os.environ["HOST"] = "0.0.0.0"
    
    if not os.getenv("LOG_LEVEL"):
        os.environ["LOG_LEVEL"] = "info"
    
    if not os.getenv("RELOAD"):
        os.environ["RELOAD"] = "true"
    
    print(f"🚀 Starting RAG API on {os.getenv('HOST')}:{os.getenv('PORT')}")
    print(f"📁 Working directory: {os.getcwd()}")
    print(f"🔄 Auto-reload: {os.getenv('RELOAD')}")
    print(f"📝 Log level: {os.getenv('LOG_LEVEL')}")

def check_dependencies():
    """Check apakah semua dependencies terpenuhi"""
    required_files = [
        "app/main.py",
        "app/api/routes.py", 
        "app/services/rag_service.py",
        "app/config/llm2.py"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Missing required files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        print("Please ensure all required files are present.")
        sys.exit(1)
    
    print("✅ All required files found")

def check_environment_variables():
    """Check environment variables yang diperlukan"""
    required_vars = [
        "GEMINI_API_KEY_1",
        "GEMINI_API_KEY_2", 
        "GEMINI_API_KEY_3",
        "GEMINI_API_KEY_4",
        "GEMINI_API_KEY_5"
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print("⚠️  Warning: Missing environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        print("Some features may not work properly.")
    else:
        print("✅ All required environment variables found")

def main():
    """Main function untuk menjalankan server"""
    try:
        # Setup environment
        setup_environment()
        
        # Check dependencies
        check_dependencies()
        
        # Check environment variables
        check_environment_variables()
        
        # Konfigurasi server
        config = uvicorn.Config(
            app="app.main:app",
            host=os.getenv("HOST", "0.0.0.0"),
            port=int(os.getenv("PORT", 8000)),
            reload=os.getenv("RELOAD", "true").lower() == "true",
            log_level=os.getenv("LOG_LEVEL", "info"),
            access_log=True,
            workers=1
        )
        
        # Jalankan server
        server = uvicorn.Server(config)
        server.run()
        
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 