# app/services/md_processor.py

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