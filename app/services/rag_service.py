# app/services/rag_service.py

import json
import os
import hashlib  # untuk menghitung hash file JSON
from pymongo import MongoClient  # untuk koneksi ke MongoDB
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import START, StateGraph
from langchain_huggingface import HuggingFaceEmbeddings
from typing_extensions import List, TypedDict
from functools import partial
from ragas import EvaluationDataset, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from app.config.rotating_llm_wrapper import RotatingLangchainLLMWrapper
from ragas.run_config import RunConfig
from ragas.metrics import LLMContextPrecisionWithoutReference ,LLMContextRecall, Faithfulness, ResponseRelevancy
from app.config.llm2 import get_next_llm

# Kelas pembungkus untuk menyimpan instance LLM agar bisa diperbarui saat rotasi
class LLMHolder:
    def __init__(self, llm):
        self.llm = llm

# Template prompt untuk menghasilkan jawaban
ANSWER_TEMPLATE = """
Anda adalah asisten yang hanya boleh menjawab pertanyaan berdasarkan potongan konteks yang disediakan di bawah ini. Ikuti instruksi berikut dengan cermat:

1. Bacalah seluruh potongan konteks.
2. Jawablah pertanyaan di akhir hanya dengan informasi yang ada di konteks.
3. Gunakan maksimal tiga kalimat yang singkat, jelas, dan informatif.
4. Jika konteks tidak cukup untuk menjawab pertanyaan, balas dengan: 
   "Maaf, saya tidak bisa menjawab pertanyaan tersebut karena belum cukup pengetahuan untuk menjawab." 
   (namun, jika ada konteks yang cukup berkaitan, berikan jawaban sesuai konteks).
5. Jangan menambahkan informasi eksternal atau menebak jawaban.
6. Hindari frasa seperti "konteks atau teks yang diberikan tidak ada".
7. Jangan menyampaikan permintaan maaf jika pertanyaan sudah terjawab dengan konteks.
8. Jika jawaban terdiri dari lebih dari satu kalimat, pastikan kalimat-kalimat tersebut saling berkaitan.
9. Hindari memberikan informasi terkait kode atau referensi yang tidak dapat diketahui oleh pengguna.
10. Akhiri setiap jawaban dengan "terima kasih sudah bertanya!".

Pengetahuan yang Anda miliki: {context}

Pertanyaan: {question}

Jawaban yang Bermanfaat:
"""

prompt_template = ChatPromptTemplate.from_messages([("user", ANSWER_TEMPLATE)])

# Template prompt untuk perbaikan query (refinement)
# REFINE_TEMPLATE = """Anda adalah asisten yang ahli dalam memperbaiki kualitas pertanyaan untuk pencarian informasi. Berikut adalah pertanyaan asli: "{question}". Perbaiki pertanyaan tersebut agar lebih umum, jelas, dan relevan untuk pencarian konteks dokumen, dengan menghilangkan frasa-frasa yang tidak membantu (misalnya, "dari survei ini"). Keluarkan pertanyaan perbaikan yang singkat dan tepat."""
REFINE_TEMPLATE = """
Anda adalah ahli dalam merumuskan ulang pertanyaan untuk pencarian dokumen dalam konteks Survei Wisatawan Nusantara. Tugas Anda adalah mengubah pertanyaan asli agar lebih umum, jelas, dan konsisten dengan istilah yang biasa digunakan dalam definisi atau konsep yang umum. Ikuti instruksi berikut:

1. Identifikasi dan hapus frasa atau kata-kata tambahan yang tidak mendukung inti pertanyaan (misalnya, "dari survei ini", "dari laporan ini").
2. Jika pertanyaan mengandung kata tunjuk, seperti ini, itu, tersebut, -nya, dan lain-lain, gantilah kata tunjuk tersebut dengan kata benda yang sesuai (contoh: ubah "contohnya" menjadi "contoh pengeluaran untuk pramuwisata").
3. Jika perlu, tambahkan preposisi atau kata penghubung agar pertanyaan menjadi lebih baku dan sesuai dengan bahasa definisi (contoh: ubah "pengeluaran pramuwisata" menjadi "pengeluaran untuk pramuwisata").
4. Pastikan inti pertanyaan tetap utuh sehingga fokus pencarian dokumen tidak terganggu.
5. Ubah pertanyaan yang menanyakan pengertian atau definisi, seperti "Apa itu ...", "Apa pengertian dari ...", "Apa yang dimaksud dengan ...", "Apa arti dari ...", "Apa maksudnya ...", dan lain-lain menjadi "Apa definisi dari ...".
6. Jika pertanyaan pertanyaan menanyakan terkait definisi atau pengertian suatu istilah, kembangkan pertanyaan tersebut untuk juga memberikan opsi tentang pertanyaan contoh atau komponen. Misalnya, pertanyaan asli: "Apa definisi dari wisata kuliner?" dapat diubah menjadi "apa definisi dari wisata kuliner atau apa saja yang termasuk dalam wisata kuliner?".
7. Tetap pertahankan bentuk pertanyaan aslinya, seperti "apa yang dimaksud dengan" atau "apa itu" atau "apa pengertian dari" atau bentuk lainnya.
8. Sebisa mungkin, pertanyaan yang dihasilkan harus sangat mendekati dengan pertanyaan asli.
9. Pastikan Anda tidak mengubah istilah atau kata kunci yang ditanyakan dalam pertanyaan asli dengan sinonim atau lainnya.
10. Jika ada pertanyaan yang tidak lengkap atau sempurna, lengkapi pertanyaan tersebut agar menjadi pertanyaan yang lebih jelas dan spesifik. Misalnya, ubah "apa yang termasuk dalam wisata kuliner" menjadi "apa saja yang termasuk dalam wisata kuliner?".

Contoh:
- Jika pertanyaan asli adalah "Apa pengertian dari bekerja dari survei ini?", keluarkan "Apa pengertian dari bekerja?".
- Jika pertanyaan asli adalah "Apa itu pengeluaran pramusaji?", keluarkan "Apa itu pengeluaran untuk pramusaji?".

Pertanyaan asli: "{question}"

Hasilkan pertanyaan yang telah direformulasi dengan menjalankan semua instruksi di atas satu per satu secara optimal untuk pencarian dokumen:
"""


refine_prompt_template = ChatPromptTemplate.from_messages([("user", REFINE_TEMPLATE)])


# Helper function untuk memanggil LLM secara aman dengan rotasi API key bila terjadi error 429
async def safe_llm_invoke(llm_holder: LLMHolder, messages):
    attempts = 0
    from app.config.llm2 import gemini_keys  # Mengambil key dari konfigurasi
    max_attempts = len(gemini_keys)
    
    while attempts < max_attempts:
        try:
            return await llm_holder.llm.ainvoke(messages)
        except Exception as e:
            error_message = str(e)
            if "429" in error_message:
                attempts += 1
                # Rotasi LLM secara asinkron
                llm_holder.llm = await get_next_llm()
            else:
                raise e
    raise Exception("All Gemini API keys exhausted due to rate limits.")

# Definisikan state untuk alur RAG
# class State(TypedDict):
#     question: str
#     context: List[Document]
#     answer: str

# Mendefinisikan tipe dokumen hasil retrieval yang baru
class RetrievedDocument(TypedDict):
    document: Document
    similarity_score: float

# Definisikan state untuk alur RAG dengan konteks berupa list RetrievedDocument
class State(TypedDict):
    question: str
    context: List[RetrievedDocument]
    answer: str



async def refine_query(state: State, llm_holder: LLMHolder) -> State:
    """Memperbaiki query menggunakan LLM dengan penanganan error dan rotasi API key."""
    messages = refine_prompt_template.invoke({"question": state["question"]})
    refined_result = await safe_llm_invoke(llm_holder, messages)
    state["question"] = refined_result.content.strip()
    return state

# def retrieve(state: State, vector_store):
#     top_k = 10
#     retrieved_docs = vector_store.similarity_search(state["question"], k=top_k)
#     return {"context": retrieved_docs}

# def retrieve(state: State, vector_store):
#     top_k = 10
#     # Mengambil dokumen beserta skor similarity
#     retrieved_docs_with_scores = vector_store.similarity_search_with_score(state["question"], k=top_k)
#     # Mengemas hasil dalam format yang baru: dictionary dengan keys 'document' dan 'similarity_score'
#     retrieved_docs = [{"document": doc, "similarity_score": score} for doc, score in retrieved_docs_with_scores]
#     return {"context": retrieved_docs}

def retrieve(state: State, vector_store):
    top_k = 10
    threshold = 0.7
    # Mengambil dokumen beserta skor similarity
    retrieved_docs_with_scores = vector_store.similarity_search_with_score(state["question"], k=top_k)
    # Hanya masukkan dokumen dengan similarity score >= threshold
    retrieved_docs = [
        {"document": doc, "similarity_score": score} 
        for doc, score in retrieved_docs_with_scores 
        if score >= threshold
    ]
    return {"context": retrieved_docs}

# async def generate(state: State, llm_holder: LLMHolder):
#     docs_content = "\n\n".join(doc.page_content for doc in state["context"])
#     messages = prompt_template.invoke({"question": state["question"], "context": docs_content})
#     response = await safe_llm_invoke(llm_holder, messages)
#     return {"answer": response.content}

async def generate(state: State, llm_holder: LLMHolder):
    # Menggabungkan isi dokumen dan menambahkan skor similarity untuk setiap dokumen
    docs_content = "\n\n".join(
        f"Dokumen {i+1}: {doc_entry['document'].page_content} (Similarity: {doc_entry['similarity_score']:.4f})"
        for i, doc_entry in enumerate(state["context"])
    )
    messages = prompt_template.invoke({"question": state["question"], "context": docs_content})
    response = await safe_llm_invoke(llm_holder, messages)
    return {"answer": response.content}



def create_rag_chain(vector_store, llm_holder: LLMHolder):
    graph_builder = StateGraph(State)

    # Node asynchronous untuk refine_query
    async def refine_node(state):
        return await refine_query(state, llm_holder)

    # Node synchronous untuk retrieval
    def retrieve_node(state):
        return retrieve(state, vector_store)

    # Node asynchronous untuk generate
    async def generate_node(state):
        return await generate(state, llm_holder)

    graph_builder.add_node("refine", refine_node)
    graph_builder.add_node("retrieve", retrieve_node)
    graph_builder.add_node("generate", generate_node)

    # Alur chain: START -> refine -> retrieve -> generate
    graph_builder.add_edge(START, "refine")
    graph_builder.add_edge("refine", "retrieve")
    graph_builder.add_edge("retrieve", "generate")

    graph = graph_builder.compile()
    return graph

async def evaluate_rag_system(vector_store, llm_holder: LLMHolder, json_file_path: str):
    """
    Fungsi untuk menyiapkan data evaluasi, menjalankan RAG untuk setiap query,
    mengevaluasi performa sistem, dan mengunggah hasil evaluasi ke dashboard ragas.
    
    Proses:
    1. Membaca file JSON dan menghitung hash isinya.
    2. Mengecek ke MongoDB apakah file JSON dengan hash tersebut dan dataset_list sudah tersimpan.
       - Jika ada, langsung gunakan dataset_list dari database.
       - Jika belum, proses pembentukan dataset_list dilakukan dan kemudian disimpan ke database.
    """
    if not os.path.isfile(json_file_path):
        raise FileNotFoundError(f"File {json_file_path} tidak ditemukan.")

    # Baca seluruh konten file JSON dan hitung hash-nya
    with open(json_file_path, 'r', encoding='utf-8') as f:
        json_content = f.read()
    file_hash = hashlib.md5(json_content.encode('utf-8')).hexdigest()

    # Koneksi ke MongoDB (sesuaikan URI dan nama database sesuai konfigurasi)
    mongodb_uri = os.getenv("MONGODB_URI")
    mongodb_db_name = os.getenv("MONGODB_DB_NAME")
    client = MongoClient(mongodb_uri)
    db = client[mongodb_db_name]
    evaluation_collection = db["dataset_evaluations"]

    # Cek apakah file JSON dan dataset_list sudah tersimpan
    record = evaluation_collection.find_one({"file_hash": file_hash})
    if record and "dataset_list" in record:
        # Gunakan dataset_list dari database jika sudah ada
        dataset_list = record["dataset_list"]
    else:
        # Jika tidak ada, parsing JSON dan bangun dataset_list
        try:
            sample_data = json.loads(json_content)
        except json.JSONDecodeError as e:
            raise ValueError(f"Error decoding JSON from {json_file_path}: {e}")

        dataset_list = []
        for data in sample_data:
            query = data["user_input"]
            reference = data["ground_truth"]
            state = {"question": query}
            state = await refine_query(state, llm_holder)
            retrieved = retrieve(state, vector_store)
            state.update(retrieved)
            generated = await generate(state, llm_holder)
            state.update(generated)
            # dataset_list.append({
            #     "user_input": query,
            #     "retrieved_contexts": [doc.page_content for doc in state["context"]],
            #     "response": state["answer"],
            #     "reference": reference
            # })
            dataset_list.append({
                "user_input": query,
                # Akses konten dokumen dari key 'document' karena format retrieval telah berubah
                "retrieved_contexts": [doc_entry["document"].page_content for doc_entry in state["context"]],
                "response": state["answer"],
                "reference": reference
            })

        # Simpan dataset_list dan informasi file ke MongoDB
        evaluation_collection.insert_one({
            "file_hash": file_hash,
            "json_file_path": json_file_path,
            "dataset_list": dataset_list
        })

    # Proses evaluasi menggunakan dataset_list yang ada
    eval_dataset = EvaluationDataset.from_list(dataset_list)
    evaluator_llm = RotatingLangchainLLMWrapper(llm_holder)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    evaluator_embeddings = LangchainEmbeddingsWrapper(embeddings)
    # evaluator_llm = LangchainLLMWrapper(llm)
    result = evaluate(
        dataset=eval_dataset,
        metrics=[LLMContextPrecisionWithoutReference(), LLMContextRecall(), Faithfulness(), ResponseRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings)],
        llm=evaluator_llm,
        run_config=RunConfig(timeout=720,max_retries=15,max_workers=2)
    )
    upload_response = result.upload()

    return {"evaluation_result": result, "upload_response": upload_response}

# async def evaluate_rag_system(vector_store, llm_holder: LLMHolder, json_file_path: str):
#     """
#     Fungsi untuk menyiapkan data evaluasi, menjalankan RAG untuk setiap query,
#     mengevaluasi performa sistem, dan mengunggah hasil evaluasi ke dashboard ragas.
#     """
#     if not os.path.isfile(json_file_path):
#         raise FileNotFoundError(f"File {json_file_path} tidak ditemukan.")

#     with open(json_file_path, 'r', encoding='utf-8') as f:
#         try:
#             sample_data = json.load(f)
#         except json.JSONDecodeError as e:
#             raise ValueError(f"Error decoding JSON from {json_file_path}: {e}")

#     dataset_list = []

#     for data in sample_data:
#         query = data["user_input"]
#         reference = data["ground_truth"]
#         state = {"question": query}
#         state = await refine_query(state, llm_holder)
#         retrieved = retrieve(state, vector_store)
#         state.update(retrieved)
#         generated = await generate(state, llm_holder)
#         state.update(generated)
#         dataset_list.append({
#             "user_input": query,
#             "retrieved_contexts": [doc.page_content for doc in state["context"]],
#             "response": state["answer"],
#             "reference": reference
#         })

#     eval_dataset = EvaluationDataset.from_list(dataset_list)
#     evaluator_llm = LangchainLLMWrapper(llm_holder.llm)

#     result = evaluate(
#         dataset=eval_dataset,
#         metrics=[LLMContextRecall(), Faithfulness(), FactualCorrectness()],
#         llm=evaluator_llm
#     )

#     upload_response = result.upload()

#     return {"evaluation_result": result, "upload_response": upload_response}


# import json
# import os
# from langchain_core.documents import Document
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.runnables import RunnableLambda
# from langchain_core.language_models import BaseChatModel
# from langchain_community.vectorstores import MongoDBAtlasVectorSearch
# from langchain_core.documents import Document
# from langgraph.graph import START, StateGraph
# from typing_extensions import List, TypedDict
# import asyncio
# from functools import partial
# from ragas import EvaluationDataset, evaluate
# from ragas.llms import LangchainLLMWrapper
# from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness
# from app.config.llm2 import get_current_llm

# # Define state for application

# class State(TypedDict):
#     question: str
#     context: List[Document]
#     answer: str


# ANSWER_TEMPLATE = """
# Anda adalah asisten yang hanya boleh menjawab pertanyaan berdasarkan potongan konteks yang disediakan di bawah ini. Ikuti instruksi berikut dengan cermat:

# 1. Bacalah seluruh potongan konteks.
# 2. Jawablah pertanyaan di akhir hanya dengan informasi yang ada di konteks.
# 3. Gunakan maksimal tiga kalimat yang singkat, jelas, dan informatif.
# 4. Jika konteks tidak cukup untuk menjawab pertanyaan, balas dengan: 
#    "Maaf, saya tidak bisa menjawab pertanyaan tersebut karena belum cukup pengetahuan untuk menjawab." tetapi jangan sekali-kali mengeluarkan jawaban ketidaktahuan tersebut ketika ada konteks yang cukup berkaitan.
# 5. Jangan menambahkan informasi eksternal atau menebak jawaban.
# 6. Hindari frasa seperti "konteks atau teks yang diberikan tidak ada" dan frasa sejenis lainnya.
# 7. Jangan menyampaikan permintaan maaf jika pertanyaan sudah terjawab dengan konteks.
# 8. Jika jawaban terdiri dari lebih dari satu kalimat, pastikan kalimat-kalimat tersebut saling berkaitan dan tidak bertentangan sehingga menjadi paragraf yang efektif termasuk jangan mengulang penggunakan subjek yang sama persis dalam setiap kalimatnya.
# 9. Hindari memberikan informasi terkait kode atau referensi yang tidak dapat diketahui oleh pengguna.
# 10. Akhiri setiap jawaban dengan "terima kasih sudah bertanya!".

# Pengetahuan yang Anda miliki: {context}

# Pertanyaan: {question}

# Jawaban yang Bermanfaat:
# """

# prompt_template = ChatPromptTemplate.from_messages([("user", ANSWER_TEMPLATE)])

# # Prompt template untuk *query refinement*
# # REFINE_TEMPLATE = """Anda adalah asisten yang ahli dalam memperbaiki kualitas pertanyaan untuk pencarian informasi. Berikut adalah pertanyaan asli: "{question}". Perbaiki pertanyaan tersebut agar lebih umum, jelas, dan relevan untuk pencarian konteks dokumen, dengan menghilangkan frasa-frasa yang tidak membantu (misalnya, "dari survei ini"). Keluarkan pertanyaan perbaikan yang singkat dan tepat."""
# REFINE_TEMPLATE = """
# Anda adalah ahli dalam merumuskan ulang pertanyaan untuk pencarian dokumen. Tugas Anda adalah mengubah pertanyaan asli agar lebih umum, jelas, dan konsisten dengan istilah yang biasa digunakan dalam definisi atau konsep yang umum. Ikuti instruksi berikut:

# 1. Identifikasi dan hapus frasa atau kata-kata tambahan yang tidak mendukung inti pertanyaan (misalnya, "dari survei ini", "dari laporan ini").
# 2. Jika pertanyaan mengandung kata tunjuk, seperti ini, itu, tersebut, -nya, dan lain-lain, gantilah kata tunjuk tersebut dengan kata benda yang sesuai (contoh: ubah "contohnya" menjadi "contoh pengeluaran untuk pramuwisata").
# 3. Jika perlu, tambahkan preposisi atau kata penghubung agar pertanyaan menjadi lebih baku dan sesuai dengan bahasa definisi (contoh: ubah "pengeluaran pramuwisata" menjadi "pengeluaran untuk pramuwisata").
# 4. Pastikan inti pertanyaan tetap utuh sehingga fokus pencarian dokumen tidak terganggu.
# 5. Jika pertanyaan pertanyaan menanyakan terkait definisi atau pengertian suatu istilah, kembangkan pertanyaan tersebut untuk juga memberikan opsi tentang pertanyaan contoh atau komponen. Misalnya, pertanyaan asli: "Apa definisi dari wisata kuliner?" dapat diubah menjadi "apa definisi dari wisata kuliner atau apa saja yang termasuk dalam wisata kuliner?".
# 6. Tetap pertahankan bentuk pertanyaan aslinya, seperti "apa yang dimaksud dengan" atau "apa itu" atau "apa pengertian dari" atau bentuk lainnya.
# 7. Sebisa mungkin, pertanyaan yang dihasilkan harus sangat mendekati dengan pertanyaan asli.
# 8. Pastikan Anda tidak mengubah istilah atau kata kunci yang ditanyakan dalam pertanyaan asli dengan sinonim atau lainnya.
# 9. Jika ada pertanyaan yang tidak lengkap atau sempurna, lengkapi pertanyaan tersebut agar menjadi pertanyaan yang lebih jelas dan spesifik. Misalnya, ubah "apa yang termasuk dalam wisata kuliner" menjadi "apa saja yang termasuk dalam wisata kuliner?".

# Contoh:
# - Jika pertanyaan asli adalah "Apa pengertian dari bekerja dari survei ini?", keluarkan "Apa pengertian dari bekerja?".
# - Jika pertanyaan asli adalah "Apa itu pengeluaran pramusaji?", keluarkan "Apa itu pengeluaran untuk pramusaji?".

# Pertanyaan asli: "{question}"

# Hasilkan pertanyaan yang telah direformulasi secara optimal untuk pencarian dokumen:
# """

# refine_prompt_template = ChatPromptTemplate.from_messages(
#     [("user", REFINE_TEMPLATE)])


# def refine_query(state: State, llm: BaseChatModel) -> State:
#     """Memperbaiki kualitas query yang diberikan oleh pengguna menggunakan LLM."""
#     messages = refine_prompt_template.invoke({"question": state["question"]})
#     refined_result = llm.invoke(messages)
#     # Perbarui state dengan query yang sudah diperbaiki
#     state["question"] = refined_result.content.strip()
#     return state


# def retrieve(state: State, vector_store):
#     top_k = 8
#     retrieved_docs = vector_store.similarity_search(state["question"], k=top_k)
#     return {"context": retrieved_docs}


# def generate(state: State, llm: BaseChatModel):
#     docs_content = "\n\n".join(doc.page_content for doc in state["context"])
#     messages = prompt_template.invoke(
#         {"question": state["question"], "context": docs_content})
#     response = llm.invoke(messages)
#     return {"answer": response.content}


# def create_rag_chain(vector_store, llm: BaseChatModel):
#     graph_builder = StateGraph(State)
#     # Tambahkan node untuk memperbaiki query
#     graph_builder.add_node("refine", partial(refine_query, llm=llm))
#     graph_builder.add_node(
#         "retrieve", lambda state: retrieve(state, vector_store))
#     graph_builder.add_node("generate", partial(generate, llm=llm))

#     # Definisikan alur chain: START -> refine -> retrieve -> generate
#     graph_builder.add_edge(START, "refine")
#     graph_builder.add_edge("refine", "retrieve")
#     graph_builder.add_edge("retrieve", "generate")

#     graph = graph_builder.compile()
#     return graph


# async def evaluate_rag_system(vector_store, llm: BaseChatModel, json_file_path: str):
#     """
#     Fungsi untuk menyiapkan data evaluasi, menjalankan RAG untuk setiap query,
#     mengevaluasi performa sistem, dan mengunggah hasil evaluasi ke dashboard ragas.
#     Jika terjadi error 429 (RESOURCE_EXHAUSTED) pada pemanggilan API LLM,
#     fungsi ini akan menginisialisasi ulang LLM dengan API key berikutnya.

#     Argumen:
#       - json_file_path: Lokasi lengkap file JSON yang berisi data evaluasi.
#     """
#     if not os.path.isfile(json_file_path):
#         raise FileNotFoundError(f"File {json_file_path} tidak ditemukan.")

#     with open(json_file_path, 'r', encoding='utf-8') as f:
#         try:
#             sample_data = json.load(f)
#         except json.JSONDecodeError as e:
#             raise ValueError(f"Error decoding JSON from {json_file_path}: {e}")

#     dataset_list = []

#     async def safe_refine_query(state: State, current_llm: BaseChatModel):
#         try:
#             state = refine_query(state, current_llm)
#             # return state, current_llm
#         except Exception as e:
#             if "429" in str(e):
#                 print("Rotasi API key karena error 429...")
#                 new_llm = get_current_llm()
#                 state = refine_query(state, new_llm)
#                 # return state, new_llm
#                 continue
#             raise
#         return state, current_llm

#     async def safe_generate(state: State, current_llm: BaseChatModel):
#         try:
#             generated = generate(state, current_llm)
#             # return generated, current_llm
#         except Exception as e:
#             if "429" in str(e):
#                 print("Rotasi API key karena error 429...")
#                 new_llm = get_current_llm()
#                 generated = generate(state, new_llm)
#                 # return generated, new_llm
#                 continue
#             raise
#         return generated, current_llm

#     # Proses setiap data evaluasi
#     for data in sample_data:
#         query = data["user_input"]
#         reference = data["ground_truth"]
#         state = {"question": query}
#         # Panggil safe_refine_query
#         state, llm = await safe_refine_query(state, llm)
#         retrieved = retrieve(state, vector_store)
#         state.update(retrieved)
#         # Panggil safe_generate
#         generated, llm = await safe_generate(state, llm)
#         state.update(generated)
#         dataset_list.append({
#             "user_input": query,
#             "retrieved_contexts": [doc.page_content for doc in state["context"]],
#             "response": state["answer"],
#             "reference": reference
#         })

#     eval_dataset = EvaluationDataset.from_list(dataset_list)
#     evaluator_llm = LangchainLLMWrapper(llm)

#     result = evaluate(
#         dataset=eval_dataset,
#         metrics=[LLMContextRecall(), Faithfulness(), FactualCorrectness()],
#         llm=evaluator_llm
#     )

#     upload_response = result.upload()

#     return {"evaluation_result": result, "upload_response": upload_response}

# INI BATASNYA

# def evaluate_rag_system(vector_store, llm: BaseChatModel, json_file_path: str):
#     """
#     Fungsi untuk menyiapkan data evaluasi, menjalankan RAG untuk setiap query,
#     mengevaluasi performa sistem, dan mengunggah hasil evaluasi ke dashboard ragas.

#     Argumen:
#     - json_file_path: Lokasi lengkap file JSON yang berisi data evaluasi.
#     """
#     # Memastikan file JSON ada
#     if not os.path.isfile(json_file_path):
#         raise FileNotFoundError(f"File {json_file_path} tidak ditemukan.")

#     # Membaca konten dari file JSON
#     with open(json_file_path, 'r', encoding='utf-8') as f:
#         try:
#             sample_data = json.load(f)  # Memuat data dari file JSON
#         except json.JSONDecodeError as e:
#             raise ValueError(f"Error decoding JSON from {json_file_path}: {e}")

#     dataset_list = []

#     # Untuk setiap data evaluasi, jalankan alur RAG
#     for data in sample_data:
#         query = data["user_input"]
#         reference = data["ground_truth"]
#         state = {"question": query}
#         state = refine_query(state, llm)
#         retrieved = retrieve(state, vector_store)
#         state.update(retrieved)
#         generated = generate(state, llm)
#         state.update(generated)
#         dataset_list.append({
#             "user_input": query,
#             "retrieved_contexts": [doc.page_content for doc in state["context"]],
#             "response": state["answer"],
#             "reference": reference
#         })

#     # Buat evaluation dataset dari list data evaluasi
#     eval_dataset = EvaluationDataset.from_list(dataset_list)
#     evaluator_llm = LangchainLLMWrapper(llm)

#     result = evaluate(
#         dataset=eval_dataset,
#         metrics=[LLMContextRecall(), Faithfulness(), FactualCorrectness()],
#         llm=evaluator_llm
#     )

#     upload_response = result.upload()

#     return {"evaluation_result": result, "upload_response": upload_response}

# from langchain_core.documents import Document
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.runnables import RunnableLambda
# from langchain_core.language_models import BaseChatModel
# from langchain_community.vectorstores import MongoDBAtlasVectorSearch
# from langchain_core.documents import Document
# from langgraph.graph import START, StateGraph
# from typing_extensions import List, TypedDict
# import asyncio
# from functools import partial

# # Define state for application


# class State(TypedDict):
#     question: str
#     context: List[Document]
#     answer: str


# # **PROMPT TEMPLATE**
# # template = """Gunakan potongan konteks berikut untuk menjawab pertanyaan di akhir.
# # Jika Anda tidak tahu jawabannya, katakan saja bahwa Anda tidak tahu, jangan mencoba membuat jawaban.
# # Gunakan maksimal tiga kalimat dan buat jawaban se-singkat mungkin. Pastikan jawaban Anda merupakan kalimat yang efektif dan informatif.
# # Anda hanya didesain untuk menjawab pertanyaan terkait konsep dan definisi secara eksplisit yang dijelaskan dalam konteks sehingga tidak dapat menjawab pertanyaan dengan tingkat reasoning yang lebih tinggi dan berikan penjelasan bahwa anda tidak bisa melakukannya.
# # Jangan gunakan istilah "konteks atau teks yang diberikan tidak ada" saat merespons pertanyaan yang tidak dapat dijawab tetapi gunakan istilah "Maaf, saya tidak bisa menjawab pertanyaan tersebut karena belum cukup pengetahuan untuk menjawab".
# # Selalu ucapkan "terima kasih sudah bertanya!" di akhir jawaban.

# # Pengetahuan yang Anda miliki: {context}

# # Pertanyaan: {question}

# # Jawaban yang Bermanfaat:"""

# template = """
# Anda adalah asisten yang hanya boleh menjawab pertanyaan berdasarkan potongan konteks yang diberikan di bawah ini.
# Instruksi:
# 1. Bacalah potongan konteks dengan cermat.
# 2. Jawablah pertanyaan di akhir hanya menggunakan informasi yang tersedia dalam konteks.
# 3. Jika konteks tidak memadai untuk menjawab pertanyaan, balas dengan: "Maaf, saya tidak bisa menjawab pertanyaan tersebut karena belum cukup pengetahuan untuk menjawab." Jangan mencoba membuat jawaban atau menebak.
# 4. Berikan jawaban dengan maksimal tiga kalimat yang ringkas, jelas, dan informatif.
# 5. Hindari penambahan informasi eksternal atau interpretasi yang mendalam; jika pertanyaan memerlukan reasoning yang lebih tinggi, jelaskan bahwa batasan tersebut ada.
# 6. Akhiri setiap jawaban dengan "terima kasih sudah bertanya!".

# Potongan konteks:
# {context}

# Pertanyaan:
# {question}

# Jawaban yang Bermanfaat:
# """

# prompt_template = ChatPromptTemplate.from_messages([("user", template)])

# # Define application steps


# def retrieve(state: State, vector_store):
#     top_k = 10
#     retrieved_docs = vector_store.similarity_search(state["question"], k=top_k)
#     return {"context": retrieved_docs}

# # Modifikasi generate untuk menerima parameter llm
# def generate(state: State, llm: BaseChatModel):
#     docs_content = "\n\n".join(doc.page_content for doc in state["context"])
#     messages = prompt_template.invoke(
#         {"question": state["question"], "context": docs_content})
#     response = llm.invoke(messages)
#     return {"answer": response.content}

# # Modifikasi create_rag_chain untuk menerima llm


# def create_rag_chain(vector_store, llm: BaseChatModel):
#     graph_builder = StateGraph(State)
#     graph_builder.add_node(
#         "retrieve", lambda state: retrieve(state, vector_store))
#     graph_builder.add_node("generate", partial(
#         generate, llm=llm))  # Gunakan partial
#     graph_builder.add_edge(START, "retrieve")
#     graph_builder.add_edge("retrieve", "generate")
#     graph = graph_builder.compile()
#     return graph
