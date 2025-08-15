# 🐍 Setup Virtual Environment - Wisnus RAG API

Panduan lengkap untuk membuat virtual environment dan menginstal semua dependensi yang diperlukan.

## 📋 Prerequisites

Sebelum memulai, pastikan Anda memiliki:
- **Python 3.8+** terinstal di sistem
- **pip** (package installer) terinstal
- **Git** (untuk clone repository)

## 🔍 Cek Versi Python

Pertama, pastikan Python sudah terinstal dengan benar:

```bash
# Cek versi Python
python --version
# atau
python3 --version

# Cek versi pip
pip --version
# atau
pip3 --version
```

## 🚀 Langkah-langkah Setup Virtual Environment

### 1. Buka Terminal/Command Prompt

```bash
# Windows (Command Prompt atau PowerShell)
cd /d/Kuliah/Semester\ 7/Koding/wisnus-rag-api

# Linux/Mac
cd /path/to/wisnus-rag-api
```

### 2. Buat Virtual Environment

```bash
# Menggunakan python
python -m venv venv

# Atau menggunakan python3 (Linux/Mac)
python3 -m venv venv
```

**Penjelasan:**
- `venv` adalah nama folder virtual environment (bisa diganti dengan nama lain)
- Virtual environment akan membuat folder `venv/` yang berisi Python interpreter terpisah

### 3. Aktifkan Virtual Environment

**Untuk Windows:**
```bash
# Command Prompt
venv\Scripts\activate

# PowerShell
venv\Scripts\Activate.ps1

# Git Bash
source venv/Scripts/activate
```

**Untuk Linux/Mac:**
```bash
source venv/bin/activate
```

**Indikator berhasil:**
- Prompt terminal akan berubah menjadi `(venv) C:\path\to\project>`
- Python interpreter akan menggunakan versi dari virtual environment

### 4. Upgrade pip (Opsional tapi Direkomendasikan)

```bash
# Setelah virtual environment aktif
pip install --upgrade pip
```

### 5. Install Dependencies

```bash
# Install semua package dari requirements.txt
pip install -r requirements.txt
```

**Jika ada error, coba:**
```bash
# Install dengan verbose output untuk debugging
pip install -r requirements.txt -v

# Atau install satu per satu jika ada konflik
pip install fastapi uvicorn motor python-dotenv
pip install langchain langchain-google-genai langchain-mongodb
pip install google-generativeai huggingface-hub
```

### 6. Verifikasi Instalasi

```bash
# Cek package yang terinstal
pip list

# Cek versi Python yang digunakan
python --version

# Test import package utama
python -c "import fastapi; print('FastAPI OK')"
python -c "import uvicorn; print('Uvicorn OK')"
python -c "import motor; print('Motor OK')"
```

## 🔧 Troubleshooting

### Masalah Umum dan Solusi

#### 1. Error "python not found"
```bash
# Pastikan Python terinstal dan ada di PATH
# Windows: Install dari python.org
# Linux: sudo apt install python3 python3-pip
# Mac: brew install python
```

#### 2. Error "pip not found"
```bash
# Install pip
python -m ensurepip --upgrade
# atau
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python get-pip.py
```

#### 3. Error saat install package
```bash
# Update pip terlebih dahulu
pip install --upgrade pip

# Install dengan --user flag
pip install --user -r requirements.txt

# Atau gunakan conda sebagai alternatif
conda create -n wisnus-rag python=3.11
conda activate wisnus-rag
pip install -r requirements.txt
```

#### 4. Permission Error (Linux/Mac)
```bash
# Gunakan sudo jika diperlukan
sudo pip install -r requirements.txt

# Atau gunakan virtual environment (direkomendasikan)
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### 5. Package Version Conflicts
```bash
# Install dengan --no-deps untuk menghindari konflik
pip install --no-deps -r requirements.txt

# Atau install manual satu per satu
pip install fastapi==0.115.13
pip install uvicorn==0.34.3
pip install motor==3.7.1
# ... dst
```

## 📦 Dependencies yang Diperlukan

Berdasarkan `requirements.txt`, aplikasi memerlukan:

### Core Dependencies:
- **FastAPI** - Web framework
- **Uvicorn** - ASGI server
- **Motor** - MongoDB async driver
- **Python-dotenv** - Environment variables

### AI/ML Dependencies:
- **LangChain** - RAG framework
- **Google Generative AI** - Gemini API
- **Hugging Face Hub** - Embeddings
- **Ragas** - Evaluation framework

### Utility Dependencies:
- **Pydantic** - Data validation
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing

## 🎯 Langkah Selanjutnya

Setelah virtual environment siap:

### 1. Setup Environment Variables
```bash
# Copy file contoh
cp env.example .env

# Edit file .env dengan API keys Anda
# GEMINI_API_KEY_1=your_key_here
# HUGGING_FACE_HUB_TOKEN=your_token_here
```

### 2. Test Aplikasi
```bash
# Jalankan aplikasi
python run.py

# Atau dengan uvicorn langsung
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 3. Akses API
- **API Docs:** http://localhost:8000/docs
- **Health Check:** http://localhost:8000/api/rag/health

## 🔄 Deactivate Virtual Environment

Ketika selesai bekerja:

```bash
# Nonaktifkan virtual environment
deactivate
```

## 📝 Tips Tambahan

### 1. Menggunakan requirements-dev.txt (jika ada)
```bash
# Install development dependencies
pip install -r requirements-dev.txt
```

### 2. Export Dependencies
```bash
# Export dependencies yang terinstal
pip freeze > requirements-current.txt
```

### 3. Clean Install
```bash
# Hapus dan buat ulang virtual environment
rm -rf venv/  # Linux/Mac
# atau
rmdir /s venv  # Windows

python -m venv venv
source venv/bin/activate  # Linux/Mac
# atau
venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### 4. Menggunakan Poetry (Alternatif)
```bash
# Install Poetry
pip install poetry

# Setup project
poetry install

# Run aplikasi
poetry run python run.py
```

## ✅ Checklist Setup

- [ ] Python 3.8+ terinstal
- [ ] Virtual environment dibuat
- [ ] Virtual environment diaktifkan
- [ ] pip diupgrade
- [ ] Dependencies terinstal
- [ ] Environment variables dikonfigurasi
- [ ] Aplikasi bisa dijalankan
- [ ] API docs bisa diakses

## 🆘 Getting Help

Jika mengalami masalah:

1. **Cek logs error** dengan detail
2. **Verifikasi versi Python** (3.8+)
3. **Pastikan virtual environment aktif**
4. **Cek koneksi internet** untuk download packages
5. **Gunakan verbose mode** untuk debugging

```bash
# Debug mode
pip install -r requirements.txt -v

# Check Python path
which python  # Linux/Mac
where python  # Windows
``` 