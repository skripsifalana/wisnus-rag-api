# Wisnus RAG API

Wisnus RAG API adalah sebuah sistem *Question-Answering* berbasis **Retrieval-Augmented Generation (RAG)** yang dirancang untuk menjawab pertanyaan seputar pariwisata di Indonesia. API ini memanfaatkan data dari "Buku Pedoman Survei Digital Wisatawan Nusantara" sebagai basis pengetahuannya.

Sistem ini dibangun menggunakan Python dengan framework FastAPI dan mengintegrasikan model bahasa canggih dari Google (Gemini) serta database vektor MongoDB Atlas untuk pencarian informasi yang relevan.

## ✨ Fitur Utama

-   **Retrieval-Augmented Generation (RAG)**: Menggabungkan pencarian informasi (retrieval) dari basis data dokumen dengan kemampuan generasi teks dari *Large Language Model* (LLM) untuk memberikan jawaban yang akurat dan kontekstual.
-   **Vector Store & Search**: Menggunakan MongoDB Atlas Vector Search untuk menyimpan dan mencari potongan-potongan informasi (chunks) dari dokumen sumber secara efisien berdasarkan kemiripan semantik.
-   **Integrasi LLM**: Terhubung dengan Google Gemini API untuk menghasilkan jawaban dalam bahasa alami. Dilengkapi dengan mekanisme rotasi API key untuk mengelola batas penggunaan.
-   **Pemrosesan Dokumen Markdown**: Mampu memproses dokumen sumber dalam format Markdown, membaginya menjadi bagian-bagian yang lebih kecil, dan menyimpannya ke dalam *vector store*.
-   **Evaluasi Performa dengan RAGAS**: Dilengkapi dengan sistem evaluasi menggunakan framework **RAGAS** untuk mengukur kualitas dan keandalan sistem RAG secara kuantitatif.
-   **API Endpoints**: Menyediakan antarmuka RESTful API yang mudah digunakan untuk inisialisasi sistem, mengajukan pertanyaan, dan menjalankan evaluasi.

## 🏗️ Arsitektur Sistem

Arsitektur sistem RAG ini mengikuti alur kerja berikut:

1.  **Inisialisasi (Indexing)**: Dokumen sumber (`.md`) dipecah menjadi beberapa bagian (chunks). Setiap *chunk* kemudian diubah menjadi vektor numerik (embedding) dan disimpan di MongoDB Atlas Vector Search.
2.  **Input Pengguna**: Pengguna mengirimkan pertanyaan melalui API.
3.  **Pencarian Konteks (Retrieval)**: Sistem mengubah pertanyaan pengguna menjadi vektor dan mencocokkannya dengan vektor dokumen yang ada di database untuk menemukan *chunks* yang paling relevan.
4.  **Augmentasi Prompt**: *Chunks* yang relevan digabungkan dengan pertanyaan asli untuk membentuk sebuah *prompt* yang kaya konteks.
5.  **Generasi Jawaban (Generation)**: *Prompt* yang telah diperkaya dikirim ke LLM (Google Gemini) untuk menghasilkan jawaban akhir yang koheren dan informatif.
6.  **Jawaban ke Pengguna**: Jawaban yang dihasilkan oleh LLM dikembalikan kepada pengguna.

## 📂 Struktur Proyek

```
wisnus-rag-api/
├── api/                      # Konfigurasi server (FastAPI)
│   └── index.py
├── app/                      # Logika inti aplikasi
│   ├── api/                  # Definisi endpoint API
│   │   └── routes.py
│   ├── config/               # Konfigurasi (DB, LLM, dll.)
│   │   ├── database.py
│   │   └── llm.py
│   ├── core/                 # Komponen inti RAG
│   │   └── rag_core.py
│   ├── models/               # Skema data Pydantic
│   │   └── schemas.py
│   ├── services/             # Layanan pemrosesan
│   │   └── rag_service.py
│   └── main.py               # Inisialisasi aplikasi FastAPI
├── docs/                     # Dokumen sumber (.md)
├── evaluation/               # Hasil evaluasi RAGAS (.json)
├── .env                      # File environment variables (dibuat manual)
├── requirements.txt          # Daftar dependensi Python
└── run.py                    # Skrip untuk menjalankan server
```

## 🚀 Instalasi & Setup

1.  **Clone repository ini:**
    ```bash
    git clone https://github.com/your-username/wisnus-rag-api.git
    cd wisnus-rag-api
    ```

2.  **Buat dan aktifkan virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Untuk Linux/macOS
    .\venv\Scripts\activate    # Untuk Windows
    ```

3.  **Install dependensi:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Konfigurasi Environment Variables:**
    Buat file `.env` di direktori root dan isi dengan konfigurasi berikut. Ganti nilai placeholder dengan kredensial Anda.
    ```env
    # Konfigurasi MongoDB
    MONGODB_URI="mongodb+srv://<user>:<password>@<cluster-url>/..."
    MONGODB_DB_NAME="wisnus_rag_db"
    MONGODB_COLLECTION_NAME="documents"
    MONGODB_INDEX_NAME="vector_index"

    # Konfigurasi Google Gemini API (bisa lebih dari satu)
    GEMINI_API_KEYS="YOUR_GEMINI_API_KEY_1,YOUR_GEMINI_API_KEY_2"

    # Konfigurasi HuggingFace (untuk embedding & evaluasi)
    HUGGINGFACEHUB_API_TOKEN="YOUR_HUGGINGFACE_HUB_TOKEN"
    ```

## 🏃 Cara Menjalankan

Jalankan aplikasi menggunakan skrip `run.py`:

```bash
python run.py
```

Server akan berjalan di `http://127.0.0.1:8000`. Anda dapat mengakses dokumentasi API interaktif (Swagger UI) di `http://127.0.0.1:8000/docs`.

## 🔬 Evaluasi dengan RAGAS

Kualitas dan keandalan sistem RAG ini dievaluasi secara ketat menggunakan **RAGAS**, sebuah framework yang dirancang khusus untuk mengevaluasi pipeline RAG. Evaluasi ini penting untuk memastikan bahwa jawaban yang diberikan tidak hanya relevan, tetapi juga akurat dan didasarkan pada konteks yang disediakan.

### Metrik Evaluasi

Kami menggunakan beberapa metrik kunci dari RAGAS:

-   **Faithfulness**: Mengukur sejauh mana jawaban yang dihasilkan sesuai dengan konteks yang diberikan. Metrik ini membantu memitigasi "halusinasi" dari LLM.
-   **Answer Relevancy**: Menilai relevansi jawaban terhadap pertanyaan pengguna. Jawaban mungkin faktual, tetapi tidak berguna jika tidak menjawab pertanyaan.
-   **Context Precision**: Mengevaluasi rasio sinyal terhadap kebisingan dalam konteks yang diambil. Apakah semua konteks yang diambil benar-benar relevan untuk menjawab pertanyaan?
-   **Context Recall**: Mengukur kemampuan sistem untuk mengambil semua informasi yang relevan dari basis pengetahuan untuk menjawab sebuah pertanyaan.

### Cara Menjalankan Evaluasi

Evaluasi dapat dipicu melalui endpoint API. Proses ini akan menggunakan dataset pertanyaan dan jawaban yang telah disiapkan sebelumnya.

```bash
POST /api/rag/evaluate
```

Hasil evaluasi akan disimpan sebagai file JSON di dalam direktori `evaluation/`. File ini berisi skor untuk setiap metrik pada setiap item dataset, serta skor rata-rata keseluruhan.

## 🔌 Endpoint API

-   `POST /api/rag/initialize`: Memproses dokumen di `docs/`, membuat *embeddings*, dan menyimpannya ke *vector store*.
-   `POST /api/rag/ask`: Mengajukan pertanyaan ke sistem. Request body: `{"question": "isi pertanyaan Anda"}`.
-   `POST /api/rag/evaluate`: Menjalankan pipeline evaluasi menggunakan RAGAS.
-   `GET /api/rag/health`: Memeriksa status kesehatan layanan.

## 🛠️ Teknologi yang Digunakan

-   **Framework**: FastAPI
-   **Bahasa**: Python 3.9+
-   **LLM**: Google Gemini Pro
-   **Database Vektor**: MongoDB Atlas Vector Search
-   **Orkestrasi RAG**: LangChain
-   **Evaluasi**: RAGAS
-   **Embedding Model**: HuggingFace `sentence-transformers/all-MiniLM-L6-v2`

## 📄 Lisensi

Proyek ini dilisensikan di bawah Lisensi MIT. Lihat file `LICENSE` untuk detailnya.