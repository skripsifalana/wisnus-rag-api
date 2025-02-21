# app/api/routes.py

from fastapi import APIRouter, HTTPException
from app.services.pdf_processor import SemanticPDFProcessor
from app.services.vector_store import VectorStoreInitializer
from app.services.rag_service import create_rag_chain, evaluate_rag_system, LLMHolder
from app.models.schemas import QuestionRequest
from app.config.llm2 import get_current_llm

router = APIRouter()
processor = SemanticPDFProcessor()
rag_chain = None
rag_vector_store = None
llm_holder = None

@router.post("/initialize")
async def initialize_rag():
    global rag_chain, rag_vector_store, llm_holder
    try:
        # Pastikan untuk await pemanggilan get_current_llm() agar instance yang dihasilkan bukan coroutine
        llm = await get_current_llm()
        llm_holder = LLMHolder(llm)
        
        # Proses dokumen PDF
        docs = processor.process_pdfs(mode="recursive", chunk_size=600, chunk_overlap=300)
        vector_store_initializer = VectorStoreInitializer()
        rag_vector_store = vector_store_initializer.initialize_vector_store()
        if rag_vector_store.collection.count_documents({}) == 0:
            rag_vector_store.add_documents(docs)

        # Buat RAG chain dengan vector_store dan llm_holder
        rag_chain = create_rag_chain(rag_vector_store, llm_holder)
        return {"message": "RAG system initialized"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Initialization failed: {str(e)}")


@router.post("/ask")
async def ask_question(request: QuestionRequest):
    if not rag_chain:
        raise HTTPException(status_code=400, detail="RAG not initialized")
    try:
        result = await rag_chain.ainvoke({"question": request.question})
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Processing failed: {str(e)}"
        )

@router.post("/evaluate")
async def evaluate_rag_route():
    if not (rag_chain and rag_vector_store and llm_holder):
        raise HTTPException(status_code=400, detail="RAG system not initialized")
    try:
        evaluation_result = await evaluate_rag_system(
            rag_vector_store, 
            llm_holder, 
            r"E:\Kuliah\Semester 7\Koding\wisnus-rag-api\app\testsets\testsets.json"
        )
        return evaluation_result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")


# from fastapi import APIRouter, HTTPException
# from app.services.pdf_processor import SemanticPDFProcessor
# from app.services.vector_store import VectorStoreInitializer
# from app.services.rag_service import create_rag_chain, evaluate_rag_system
# from app.models.schemas import QuestionRequest
# from app.config.llm import get_current_llm

# router = APIRouter()
# processor = SemanticPDFProcessor()

# # Variabel global untuk menyimpan RAG chain, vector store, dan LLM
# rag_chain = None
# rag_vector_store = None
# rag_llm = None

# @router.post("/initialize")
# async def initialize_rag():
#     global rag_chain, rag_vector_store, rag_llm
#     try:
#         llm = await get_current_llm()
#         docs = processor.process_pdfs(mode="recursive", chunk_size=600, chunk_overlap=300)
#         vector_store_initializer = VectorStoreInitializer()
#         vector_store = vector_store_initializer.initialize_vector_store()
#         existing_count = vector_store.collection.count_documents({})
#         if existing_count == 0:
#             vector_store.add_documents(docs)
#         rag_vector_store = vector_store
#         rag_llm = llm
#         rag_chain = create_rag_chain(vector_store, llm)
#         return {"message": "RAG system initialized"}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Initialization failed: {str(e)}")

# @router.post("/ask")
# async def ask_question(request: QuestionRequest):
#     if not rag_chain:
#         raise HTTPException(status_code=400, detail="RAG not initialized")
#     try:
#         result = rag_chain.invoke({"question": request.question})
#         return result
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

# @router.post("/evaluate")
# async def evaluate_rag_route():
#     if not (rag_chain and rag_vector_store):
#         raise HTTPException(status_code=400, detail="RAG system not initialized")
#     try:
#         evaluation_result = await evaluate_rag_system(rag_vector_store, rag_llm, r"E:\Kuliah\Semester 7\Koding\wisnus-rag-api\app\testsets\testsets.json")
#         return evaluation_result
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")


# from fastapi import APIRouter, Depends, HTTPException, Body
# from app.services.pdf_processor import SemanticPDFProcessor
# from app.services.vector_store import VectorStoreInitializer
# from app.services.rag_service import create_rag_chain
# from app.models.schemas import QuestionRequest
# from app.config.llm import get_current_llm

# router = APIRouter()
# processor = SemanticPDFProcessor()
# rag_chain = None


# @router.post("/initialize")
# async def initialize_rag():
#     global rag_chain
#     try:
#         # Inisialisasi LLM terlebih dahulu
#         llm = await get_current_llm()

#         # docs = process_pdfs()
#         # For semantic splitting with default threshold
#         # docs = processor.process_pdfs(mode="semantic")
#         # For tighter semantic chunks
#         # docs = processor.process_pdfs(mode="semantic", semantic_threshold=0.65)
#         # For traditional splitting
#         docs = processor.process_pdfs(mode="recursive", chunk_size=600, chunk_overlap=300)
#         vector_store_initializer = VectorStoreInitializer()
#         vector_store = vector_store_initializer.initialize_vector_store()
#         existing_count = vector_store.collection.count_documents({})
#         if existing_count == 0:
#             vector_store.add_documents(docs)

#         # Kirim llm ke create_rag_chain
#         rag_chain = create_rag_chain(vector_store, llm)
#         return {"message": "RAG system initialized"}
#     except Exception as e:
#         raise HTTPException(
#             status_code=500, detail=f"Initialization failed: {str(e)}")


# @router.post("/ask")
# async def ask_question(request: QuestionRequest):
#     if not rag_chain:
#         raise HTTPException(status_code=400, detail="RAG not initialized")
#     try:
#         result = rag_chain.invoke({"question": request.question})
#         return result
#     except Exception as e:
#         raise HTTPException(
#             status_code=500, detail=f"Processing failed: {str(e)}")