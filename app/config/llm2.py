# app/config/llm2.py

import os
from langchain_google_genai import ChatGoogleGenerativeAI

# Ambil API key dari environment variable dengan format GEMINI_API_KEY_1, GEMINI_API_KEY_2, ...
gemini_keys = []
i = 1
while True:
    key = os.environ.get(f"GEMINI_API_KEY_{i}")
    if not key:
        break
    gemini_keys.append(key)
    i += 1

if not gemini_keys:
    raise Exception("No Gemini API keys provided in environment variables")

current_api_key_index = 0

async def get_current_llm():
    """Mengembalikan instance ChatGoogleGenerativeAI dengan API key saat ini."""
    global current_api_key_index
    print(f"Membuat instance LLM dengan API key ke-{current_api_key_index + 1}")
    # Misal jika konstruktor membutuhkan await (atau jika ada proses inisialisasi async)
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0,
        api_key=gemini_keys[current_api_key_index]
    )

async def get_next_llm():
    """Rotasi ke API key berikutnya dan kembalikan instance LLM baru."""
    global current_api_key_index
    current_api_key_index = (current_api_key_index + 1) % len(gemini_keys)
    print(f"Merotasi LLM dengan API key ke-{current_api_key_index + 1}")
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0,
        api_key=gemini_keys[current_api_key_index]
    )
