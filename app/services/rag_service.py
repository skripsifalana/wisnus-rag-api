# app/services/rag_service.py

import json
import os
import hashlib
from pymongo import MongoClient
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import START, StateGraph
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from typing_extensions import List, TypedDict
from functools import partial
from ragas import EvaluationDataset, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from app.config.rotating_llm_wrapper import RotatingLangchainLLMWrapper
from ragas.run_config import RunConfig
from ragas.metrics import LLMContextPrecisionWithoutReference, LLMContextRecall, Faithfulness, ResponseRelevancy
from app.config.llm2 import gemini_keys, get_next_llm
from pydantic import BaseModel, Field
from langchain_core.runnables import chain
from langchain.output_parsers import PydanticOutputParser
import asyncio

# Kelas pembungkus untuk menyimpan instance LLM agar bisa diperbarui saat rotasi


class LLMHolder:
    def __init__(self, llm):
        self.llm = llm


# Template prompt untuk menghasilkan jawaban
ANSWER_TEMPLATE = """
Anda adalah asisten yang hanya boleh menjawab pertanyaan berdasarkan potongan konteks yang disediakan di bawah ini. Ikuti instruksi berikut dengan cermat:

1. Bacalah seluruh potongan konteks.
2. Pastikan Anda memberikan jawaban yang semirip mungkin dengan pengetahuan pada konteks yang diberikan.
3. Cobalah menjawab pertanyaan pengguna sebisa Anda serta jelaskan argumen Anda dengan tetap menggunakan konteks yang diberikan sehingga anda tetap memberikan jawaban dan tidak menjawab bahwa anda tidak tahu atau tidak bisa menjawab pertanyaan tersebut.
4. Hindari frasa seperti "konteks atau teks yang diberikan tidak ada".
5. Jangan menyampaikan permintaan maaf jika pertanyaan sudah terjawab dengan konteks.
6. Jika jawaban terdiri dari lebih dari satu kalimat, pastikan kalimat-kalimat tersebut saling berkaitan.
7. Hindari memberikan informasi terkait kode atau referensi yang tidak dapat diketahui oleh pengguna.
8. Jika konteks yang diberikan tidak menjelaskan pertanyaan secara spesifik, saya izinkan Anda memberikan jawaban menggunakan pengetahuan umum yang Anda miliki dan beri teks "(Sumber: Pengetahuan umum)." tetapi pastikan Anda tidak menjelaskan bahwa konteks yang dimiliki tidak menjawab pertanyaan secara spesifik.
9. Jika Anda dapat memberikan jawaban sesuai konteks yang tersedia, berikan teks "(Sumber: Badan Pusat Statistik)." pada akhir jawaban.
10. Akhiri setiap jawaban dengan "Terima kasih sudah bertanya!" tanpa membuat baris baru.
11. Jika ada teks "(Sumber: Badan Pusat Statistik)" dan "Terima kasih sudah bertanya!" pisahkan keduanya dengan ". " (titik spasi).
12. Hanya diperbolehkan menyertakan dua teks sumber saja, yaitu "(Sumber: Badan Pusat Statistik)" atau "(Sumber: Pengetahuan umum)".    

Pengetahuan yang Anda miliki: {context}

Pertanyaan: {question}

Jawaban yang Bermanfaat:
"""

prompt_template = ChatPromptTemplate.from_messages([("user", ANSWER_TEMPLATE)])


# Helper function untuk memanggil LLM secara aman dengan rotasi API key bila terjadi error 429
async def safe_llm_invoke(llm_holder: LLMHolder, messages):
    attempts = 0
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

# Fungsi helper untuk memanggil LLM secara streaming dengan rotasi API key bila terjadi error 429


async def safe_llm_invoke_stream(llm_holder: LLMHolder, messages):
    attempts = 0
    max_attempts = len(gemini_keys)
    while attempts < max_attempts:
        try:
            async for token in llm_holder.llm.ainvoke_stream(messages):
                yield token
            return
        except Exception as e:
            error_message = str(e)
            if "429" in error_message:
                attempts += 1
                llm_holder.llm = await get_next_llm()
            else:
                raise e
    raise Exception("All Gemini API keys exhausted due to rate limits.")


class RetrievedDocument(TypedDict):
    document: Document
    similarity_score: float

# Definisikan state untuk alur RAG dengan konteks berupa list RetrievedDocument


class State(TypedDict):
    question: str
    context: List[RetrievedDocument]
    answer: str


class Search(BaseModel):
    queries: List[str] = Field(
        description="Daftar query yang dihasilkan untuk memperluas pencarian",
        min_items=4,
        max_items=4
    )


async def multi_query_retrieval_chain(
    state: State,
    vector_store,
    llm_holder: LLMHolder,
    top_k: int = 3,
    similarity_threshold: float = 0.8,
    fetch_k: int = 20,
    lambda_mult: float = 0.5
):
    system_prompt = """
    Anda adalah ahli pembuatan kueri penelusuran untuk mengekstrak informasi relevan dari basis data vektor. Berdasarkan pertanyaan pengguna, buat EMPAT kueri berbeda dengan langkah berikut:
    1. Ekstrak kata kunci utama dari pertanyaan.
    2. Buat:
    - Query 1: Format "[kata kunci]?" 
        (Contoh: Dari "Apa definisi dari eko wisata dalam survei ini?" ambil "eko wisata" sehingga menjadi "Eko wisata?")
    - Query 2: Format "Apa itu [kata kunci]?" 
        (Contoh: "Apa itu eko wisata?")
    - Query 3: Format "Jelaskan tentang [kata kunci]?" 
        (Contoh: "Jelaskan tentang eko wisata?")
    - Query 4: Bentuk pertanyaan yang mendekati pertanyaan asli dengan menghilangkan kata-kata yang tidak penting dari pertanyaan asli yang dapat menganggu pencarian seperti "dalam survei ini", "sebenarnya", "coba" atau "gitu", dan lain-lain.
    3. Pastikan Anda membuat tepat EMPAT kueri.
    """

    user_prompt = """
    Buatkan empat query untuk pertanyaan berikut: "{question}".
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", user_prompt)
    ])

    structured_llm = llm_holder.llm.with_structured_output(Search)
    query_analyzer = prompt | structured_llm

    # Pemanggilan LLM dengan rotasi API key secara terus-menerus bila terjadi error 429
    while True:
        try:
            response = await query_analyzer.ainvoke(state["question"])
            break
        except Exception as e:
            error_message = str(e)
            if "429" in error_message:
                # Rotasi LLM secara asinkron untuk mengganti API key
                llm_holder.llm = await get_next_llm()
            else:
                raise e

    async def fetch_results(query):
        results = await vector_store.amax_marginal_relevance_search(
            query,
            k=top_k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult
        )
        # Modifikasi: Hapus embedding dari metadata
        processed_docs = []
        for doc in results:
            # Salin metadata tanpa atribut embedding
            filtered_metadata = {k: v for k,
                                 v in doc.metadata.items() if k != 'embedding'}

            # Buat dokumen baru dengan metadata yang sudah difilter
            processed_doc = Document(
                page_content=doc.page_content,
                metadata=filtered_metadata
            )
            processed_docs.append(processed_doc)

        return [{"document": doc, "similarity_score": 0.0} for doc in processed_docs]

    tasks = [fetch_results(query) for query in response.queries]
    results = await asyncio.gather(*tasks)
    all_results_with_scores = [doc for result in results for doc in result]

    # Proses deduplikasi berdasarkan _id pada metadata
    unique_results = []
    seen_ids = set()
    for doc_entry in all_results_with_scores:
        doc_id = doc_entry["document"].metadata.get("_id")
        if doc_id not in seen_ids:
            unique_results.append(doc_entry)
            seen_ids.add(doc_id)

    return unique_results


async def retrieve(state: State, vector_store, llm_holder: LLMHolder):
    top_k = 8
    similarity_threshold = 0.8
    retrieved_docs = await multi_query_retrieval_chain(state, vector_store, llm_holder, top_k, similarity_threshold)
    return {"context": retrieved_docs}

# Prompt untuk reranking dokumen
RERANK_PROMPT = """
Anda adalah asisten yang ahli dalam mengevaluasi relevansi dokumen berdasarkan pertanyaan pengguna.
Berikut adalah daftar dokumen yang didapatkan dengan skor kemiripan:
{documents}
Berdasarkan pertanyaan: {question}
Tolong lakukan hal berikut:
1. Hapus dokumen yang tidak relevan dengan pertanyaan.
2. Urutkan dokumen yang tersisa berdasarkan relevansi secara menurun (dokumen paling relevan di atas).
Berikan output dalam format JSON sebagai list, dimana setiap item memiliki:
- "document": isi dokumen,
- "similarity_score": skor relevansi (antara 0 dan 1)
Pastikan output JSON valid.
"""

# Definisikan model Pydantic untuk output terstruktur


class RerankDocument(BaseModel):
    """Representasi dokumen yang direrank dengan skor similaritas."""
    document: str = Field(
        description="Konten dokumen yang direrank"
    )
    similarity_score: float = Field(
        description="Skor similaritas dokumen setelah reranking"
    )


class RerankResult(BaseModel):
    """Kumpulan dokumen yang direrank."""
    reranked_documents: List[RerankDocument] = Field(
        description="Daftar dokumen yang direrank"
    )


async def rerank_node(state: State, llm_holder: LLMHolder):
    """
    Komponen reranking:
    1. Menerima state dengan daftar dokumen (context) dari retrieval.
    2. Memformat daftar dokumen beserta skor awal untuk dikirim ke LLM.
    3. LLM melakukan eliminasi dokumen yang tidak relevan dan mereranking sisanya.
    4. Output JSON diparsing dan state diperbarui dengan daftar dokumen reranked.

    Catatan: Log setiap eksekusi node ini agar dapat ditrace (misalnya melalui LangSmith).
    """
    # Buat string daftar dokumen untuk prompt reranking
    docs_str = "\n".join(
        f"Dokumen {i+1}: {doc_entry['document'].page_content} (Skor awal: {doc_entry['similarity_score']:.4f})"
        for i, doc_entry in enumerate(state["context"])
    )

    similarity_threshold = 0.7

    # Siapkan prompt untuk reranking
    system_prompt = """
    Anda adalah asisten yang ahli dalam mengevaluasi relevansi dokumen berdasarkan pertanyaan pengguna.
    Tolong lakukan hal berikut:
    1. Hapus dokumen yang tidak benar-benar relevan dengan pertanyaan sehingga dokumen-dokumen yang diurutkan adalah dokumen paling relevan.
    2. Urutkan dokumen yang tersisa berdasarkan relevansi secara menurun (dokumen paling relevan di atas).
    3. Pastikan dokumen yang Anda kembalikan minimal similarity scorenya 0.8.
    4. Pastikan Anda tidak mengembalikan respons yang kosong atau null.
    5. Perhatikan dan lakukan sesuai dengan instruksi yang diberikan di atas.
    """

    user_prompt = """
    - Pertanyaan:
    {question}
    - Dokumen untuk direrank:
    {documents}
    """

    # Gunakan LLM dengan output terstruktur
    llm_with_tools = llm_holder.llm.with_structured_output(
        RerankResult,
        include_raw=False  # Hanya kembalikan output terstruktur
    )

    # Buat prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", user_prompt)
    ])

    # Gabungkan prompt, LLM, dan output parser
    rerank_chain = prompt | llm_with_tools

    # Pemanggilan LLM dengan rotasi API key secara terus-menerus bila terjadi error 429
    while True:
        try:
            rerank_result = await rerank_chain.ainvoke({
                "question": state["question"],
                "documents": docs_str
            })
            break
        except Exception as e:
            error_message = str(e)
            if "429" in error_message:
                # Rotasi LLM secara asinkron untuk mengganti API key
                llm_holder.llm = await get_next_llm()
            else:
                raise e

    # Konversi hasil rerank ke format state semula
    new_context = []
    for doc in rerank_result.reranked_documents:
        # Lakukan filtering berdasarkan similarity score
        if doc.similarity_score < similarity_threshold:
            continue
        new_doc = Document(
            page_content=doc.document,
            metadata={}
        )
        new_context.append({
            "document": new_doc,
            "similarity_score": doc.similarity_score
        })

    # Perbarui state dengan dokumen yang direrank
    state["context"] = new_context
    return state


async def generate(state: State, llm_holder: LLMHolder):
    # Menggabungkan isi dokumen dan menambahkan skor similarity untuk setiap dokumen
    docs_content = "\n\n".join(
        f"Dokumen {i+1}: {doc_entry['document'].page_content} (Similarity: {doc_entry['similarity_score']:.4f})"
        for i, doc_entry in enumerate(state["context"])
    )
    messages = prompt_template.invoke(
        {"question": state["question"], "context": docs_content})
    response = await safe_llm_invoke(llm_holder, messages)
    return {"answer": response.content}


async def generate_stream(state: State, llm_holder: LLMHolder):
    """
    Versi streaming: Menghasilkan jawaban final dengan mengalirkan token satu per satu.
    """
    docs_content = "\n\n".join(
        f"Dokumen {i+1}: {doc_entry['document'].page_content} (Similarity: {doc_entry['similarity_score']:.4f})"
        for i, doc_entry in enumerate(state["context"])
    )
    messages = prompt_template.invoke(
        {"question": state["question"], "context": docs_content}
    )

    async for chunk in llm_holder.llm.astream(messages):
        yield {"answer": chunk.content}  # Mengalirkan setiap token


def create_rag_chain(vector_store, llm_holder: LLMHolder, streaming: bool = False):
    graph_builder = StateGraph(State)

    # Node synchronous untuk retrieval
    async def retrieve_node(state):
        return await retrieve(state, vector_store, llm_holder)

    # Node asynchronous untuk reranking
    async def rerank_chain_node(state):
        return await rerank_node(state, llm_holder)

    async def generate_node(state):
        if streaming:
            async for answer in generate_stream(state, llm_holder):
                yield answer  # Mengalirkan jawaban
        else:
            # return await generate(state, llm_holder)
            yield await generate(state, llm_holder)

    graph_builder.add_node("retrieve", retrieve_node)
    graph_builder.add_node("rerank", rerank_chain_node)
    graph_builder.add_node("generate", generate_node)

    # Atur alur: START -> retrieve -> rerank -> generate
    graph_builder.add_edge(START, "retrieve")
    graph_builder.add_edge("retrieve", "rerank")
    graph_builder.add_edge("rerank", "generate")

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
            # Proses retrieval
            retrieved = await retrieve(state, vector_store, llm_holder)
            state.update(retrieved)
            # Proses reranking
            state = await rerank_node(state, llm_holder)
            # Proses generate
            generated = await generate(state, llm_holder)
            state.update(generated)
            dataset_list.append({
                "user_input": query,
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
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001", google_api_key=os.getenv("GEMINI_API_KEY_1"))
    evaluator_embeddings = LangchainEmbeddingsWrapper(embeddings)
    result = evaluate(
        dataset=eval_dataset,
        metrics=[LLMContextPrecisionWithoutReference(), LLMContextRecall(), Faithfulness(
        ), ResponseRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings)],
        llm=evaluator_llm,
        run_config=RunConfig(timeout=720, max_retries=15, max_workers=2)
    )
    upload_response = result.upload()

    return {"evaluation_result": result, "upload_response": upload_response}
