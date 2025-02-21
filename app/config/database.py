import os
import sys
from pymongo import MongoClient
from dotenv import load_dotenv

# Muat variabel lingkungan dari .env
load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI")

def connect_db():
    try:
        client = MongoClient(MONGODB_URI)
        # Lakukan ping untuk memastikan koneksi
        client.admin.command('ping')
        print("MongoDB connected successfully")
        return client
    except Exception as e:
        print("MongoDB connection error:", e)
        sys.exit(1)

def get_db():
    db_name = os.getenv("MONGODB_DB_NAME")
    client = connect_db()
    return client[db_name]
