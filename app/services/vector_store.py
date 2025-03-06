# app/services/vector-store.py

import os
import json
from pymongo import MongoClient
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

class VectorStoreInitializer:
    def __init__(self):
        self.gemini_api_key = os.getenv("GEMINI_API_KEY_1")
        self.hugging_face_hub_token = os.getenv("HUGGING_FACE_HUB_TOKEN")
        self.mongodb_uri = os.getenv("MONGODB_URI")
        self.mongodb_db_name = os.getenv("MONGODB_DB_NAME")
        self.mongodb_collection_name = os.getenv("MONGODB_COLLECTION_NAME")
        self.index_name = "vector_index"

    def initialize_vector_store(self):
        if not all([self.gemini_api_key, self.hugging_face_hub_token, self.mongodb_uri, self.mongodb_db_name, self.mongodb_collection_name]):
            raise ValueError("Missing required environment variables")

        # Initialize embeddings
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=self.gemini_api_key)
        # Connect to MongoDB
        client = MongoClient(self.mongodb_uri)
        collection = client[self.mongodb_db_name][self.mongodb_collection_name]

        # Initialize MongoDB Atlas Vector Store
        vector_store = MongoDBAtlasVectorSearch(
            embedding=embeddings,
            collection=collection,
            index_name=self.index_name,
            relevance_score_fn="cosine",
        )
        return vector_store

# Example usage:
# vector_store_initializer = VectorStoreInitializer()
# vector_store = vector_store_initializer.initialize_vector_store()
