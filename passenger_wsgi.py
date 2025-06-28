#!/usr/bin/env python3
"""
Passenger WSGI entry point untuk cPanel deployment
Simple dan robust tanpa infinite recursion
"""

import os
import sys
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add current directory to Python path safely
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Set default environment variables
os.environ.setdefault('PORT', '8000')
os.environ.setdefault('HOST', '0.0.0.0')
os.environ.setdefault('LOG_LEVEL', 'info')

def create_fallback_app():
    """Create fallback application untuk error handling"""
    def fallback_app(environ, start_response):
        html = '''<!DOCTYPE html>
<html><head><title>Wisnus RAG API - Error</title>
<style>body{font-family:sans-serif;background:#f5f5f5;margin:40px;} .err{background:#ffebee;color:#b71c1c;padding:20px;border-radius:8px;} .info{background:#e3f2fd;color:#1976d2;padding:20px;border-radius:8px;margin-top:20px;} code{background:#eee;padding:2px 6px;border-radius:3px;}</style>
</head><body>
<h1>🚨 Wisnus RAG API - Error 500</h1>
<div class="err">
<b>Internal Server Error</b><br>
Aplikasi gagal dijalankan.<br>
</div>
<div class="info">
<b>Solusi & Troubleshooting:</b>
<ol>
<li>Pastikan <code>requirements.txt</code> sudah diinstall di virtualenv</li>
<li>Set environment variable (misal: <code>MONGODB_URI</code>, API KEY) di cPanel</li>
<li>Periksa struktur folder: <code>app/main.py</code> harus ada</li>
<li>Restart Python App di cPanel</li>
<li>Cek error log di cPanel</li>
</ol>
</div>
</body></html>'''
        start_response('500 Internal Server Error', [('Content-Type', 'text/html; charset=utf-8')])
        return [html.encode('utf-8')]
    return fallback_app

# Initialize application variable
application = None

# Try to load FastAPI application
try:
    logger.info("🔄 Loading FastAPI application...")
    
    # Import FastAPI app
    from app.main import app
    
    # Use the FastAPI app directly
    application = app
    
    logger.info("✅ FastAPI application loaded successfully")
    
except ImportError as e:
    logger.error(f"❌ Import error: {e}")
    application = create_fallback_app()
except Exception as e:
    logger.error(f"❌ Failed to load FastAPI app: {e}")
    application = create_fallback_app()

# Ensure application is callable
if not callable(application):
    logger.error("❌ Application object is not callable! Using fallback app.")
    application = create_fallback_app()

logger.info("🚀 WSGI application ready")

# Export the application for Passenger
__all__ = ['application']