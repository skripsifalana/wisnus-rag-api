# app/services/pdf_processor.py

from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

class SemanticPDFProcessor:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY_1")
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=self.api_key
        )
        
    def process_pdfs(self, directory: str = "./pdfs", 
                    mode: str = "semantic",
                    semantic_threshold: float = 0.8,
                    chunk_size: int = 1000,
                    chunk_overlap: int = 200):
        """
        Process PDFs with either semantic or recursive splitting
        
        Args:
            directory: PDF directory path
            mode: 'semantic' or 'recursive'
            semantic_threshold: Embedding similarity threshold for splits (0-1)
            chunk_size: Fallback size for recursive splitting
            chunk_overlap: Fallback overlap for recursive splitting
        """
        if mode == "semantic" and not self.api_key:
            raise ValueError("Gemini API key required for semantic splitting")

        pdf_dir = Path(directory)
        self._validate_directory(pdf_dir)

        documents = self._load_pdfs(pdf_dir)
        return self._split_documents(documents, mode, semantic_threshold, chunk_size, chunk_overlap)

    def _validate_directory(self, pdf_dir):
        if not pdf_dir.exists() or not pdf_dir.is_dir():
            raise ValueError(f"Invalid directory: {pdf_dir}")

    def _load_pdfs(self, pdf_dir):
        documents = []
        for pdf_path in pdf_dir.glob("*.pdf"):
            loader = PyPDFLoader(str(pdf_path))
            docs = loader.load_and_split()
            documents.extend(docs)
        return documents

    def _split_documents(self, documents, mode, threshold, chunk_size, chunk_overlap):
        if mode == "semantic":
            return self._semantic_split(documents, threshold)
        return self._recursive_split(documents, chunk_size, chunk_overlap)

    def _semantic_split(self, documents, threshold):
        splitter = SemanticChunker(
            embeddings=self.embeddings,
            breakpoint_threshold_type='percentile',
            breakpoint_threshold_amount=threshold,
            add_start_index=True
        )
        return splitter.split_documents(documents)

    def _recursive_split(self, documents, chunk_size, chunk_overlap):
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        return splitter.split_documents(documents)

# if __name__ == "__main__":
#     processor = SemanticPDFProcessor()
#     processed_docs = processor.process_pdfs(
#         mode="semantic",
#         semantic_threshold=0.78
#     )
#     print(f"Total semantic chunks created: {len(processed_docs)}")
#     print("Sample chunk:", processed_docs[0].page_content[:200] + "...")

# from pathlib import Path
# from langchain_community.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter

# def process_pdfs(directory: str = "./pdfs"):
#     """
#     Memuat dan memproses semua file PDF dalam direktori tertentu.

#     Args:
#         directory (str): Path ke direktori tempat file PDF disimpan.

#     Returns:
#         List[Document]: Daftar dokumen yang telah diproses dan dipecah menjadi chunk.
#     """
#     pdf_dir = Path(directory)
#     if not pdf_dir.exists() or not pdf_dir.is_dir():
#         raise ValueError(f"Direktori '{directory}' tidak ditemukan atau bukan direktori yang valid.")

#     documents = []
#     for pdf_path in pdf_dir.glob("*.pdf"):
#         loader = PyPDFLoader(str(pdf_path))
#         docs = loader.load()
#         documents.extend(docs)
    
#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=600,
#         chunk_overlap=200,
#     )
    
#     return splitter.split_documents(documents)

# if __name__ == "__main__":
#     processed_docs = process_pdfs()
#     print(f"Total chunks created: {len(processed_docs)}")
