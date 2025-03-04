# app/services/pdf_processor.py

from pathlib import Path
from dotenv import load_dotenv
import os
import re

from langchain_core.documents import Document

# Load environment variables
load_dotenv()


class MarkdownProcessor:
    """
    Processor yang melakukan splitting dokumen Markdown berdasarkan paragraf.
    """
    def __init__(self):
        pass

    def process_markdowns(self, directory: str = "./docs"):
        """
        Proses dokumen Markdown dalam direktori dengan melakukan splitting berdasarkan paragraf.
        
        Args:
            directory: Path ke folder yang berisi file Markdown.
            
        Returns:
            List[Document]: Dokumen-dokumen yang telah di-split berdasarkan paragraf.
        """
        md_dir = Path(directory)
        self._validate_directory(md_dir)
        documents = self._load_and_split_markdowns(md_dir)
        return documents

    def _validate_directory(self, md_dir: Path):
        if not md_dir.exists() or not md_dir.is_dir():
            raise ValueError(f"Invalid directory: {md_dir}")

    def _load_and_split_markdowns(self, md_dir: Path):
        all_docs = []
        for md_path in md_dir.glob("*.md"):
            text = self._read_markdown(str(md_path))
            paragraphs = self._split_text(text)
            # Buat Document untuk setiap paragraf
            for i, para in enumerate(paragraphs):
                doc = Document(
                    page_content=para.strip(),
                    metadata={
                        "source": str(md_path),
                        "chunk_id": f"{md_path.stem}_chunk_{i+1}"
                    }
                )
                all_docs.append(doc)
        return all_docs

    def _read_markdown(self, md_path: str) -> str:
        with open(md_path, "r", encoding="utf-8") as f:
            return f.read()

    def _split_text(self, text: str):
        """
        Melakukan splitting dokumen Markdown berdasarkan paragraf.
        Diasumsikan bahwa setiap paragraf dipisahkan oleh satu baris kosong.
        """
        # Regex untuk memisahkan berdasarkan baris kosong
        paragraphs = re.split(r'\n\s*\n', text)
        # Hapus entri kosong
        return [para for para in paragraphs if para.strip()]


# Contoh penggunaan:
# if __name__ == "__main__":
#     processor = CustomPDFProcessor()
#     docs = processor.process_pdfs("./pdfs")
#     print(f"Total dokumen yang dihasilkan: {len(docs)}")


# # app/services/pdf_processor.py

# from pathlib import Path
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_experimental.text_splitter import SemanticChunker
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain_huggingface import HuggingFaceEmbeddings
# from dotenv import load_dotenv
# import os

# # Load environment variables
# load_dotenv()

# class SemanticPDFProcessor:
#     def __init__(self):
#         self.gemini_api_key = os.getenv("GEMINI_API_KEY_1")
#         self.hugging_face_hub_token = os.getenv("HUGGING_FACE_HUB_TOKEN")
#         self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=self.gemini_api_key)
#         # self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        
        
#     def process_pdfs(self, directory: str = "./pdfs", 
#                     mode: str = "semantic",
#                     semantic_threshold: float = 95.0,
#                     chunk_size: int = 1000,
#                     chunk_overlap: int = 200,
#                     min_chunk_size: int = None):
#         """
#         Process PDFs with either semantic or recursive splitting
        
#         Args:
#             directory: PDF directory path
#             mode: 'semantic' or 'recursive'
#             semantic_threshold: Embedding similarity threshold for splits (0-1)
#             chunk_size: Fallback size for recursive splitting
#             chunk_overlap: Fallback overlap for recursive splitting
#         """
#         if mode == "semantic" and not self.hugging_face_hub_token:
#             raise ValueError("Hugging Face Hub Token required for semantic splitting")

#         pdf_dir = Path(directory)
#         self._validate_directory(pdf_dir)

#         documents = self._load_pdfs(pdf_dir)
#         return self._split_documents(documents, mode, semantic_threshold, chunk_size, chunk_overlap, min_chunk_size)

#     def _validate_directory(self, pdf_dir):
#         if not pdf_dir.exists() or not pdf_dir.is_dir():
#             raise ValueError(f"Invalid directory: {pdf_dir}")

#     def _load_pdfs(self, pdf_dir):
#         documents = []
#         for pdf_path in pdf_dir.glob("*.pdf"):
#             loader = PyPDFLoader(str(pdf_path))
#             docs = loader.load_and_split()
#             documents.extend(docs)
#         return documents

#     def _split_documents(self, documents, mode, threshold, chunk_size, chunk_overlap, min_chunk_size):
#         if mode == "semantic":
#             return self._semantic_split(documents, threshold, min_chunk_size)
#         return self._recursive_split(documents, chunk_size, chunk_overlap)

#     def _semantic_split(self, documents, threshold, min_chunk_size):
#         splitter = SemanticChunker(
#             embeddings=self.embeddings,
#             breakpoint_threshold_type='percentile',
#             breakpoint_threshold_amount=threshold,
#             min_chunk_size=min_chunk_size,
#             add_start_index=True
#         )
#         return splitter.split_documents(documents)

#     def _recursive_split(self, documents, chunk_size, chunk_overlap):
#         from langchain.text_splitter import RecursiveCharacterTextSplitter
#         splitter = RecursiveCharacterTextSplitter(
#             chunk_size=chunk_size,
#             chunk_overlap=chunk_overlap
#         )
#         return splitter.split_documents(documents)

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
