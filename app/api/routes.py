# app/api/routes.py

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from app.services.md_processor import MarkdownProcessor
from app.services.vector_store import VectorStoreInitializer
from app.services.rag_service import create_rag_chain, evaluate_rag_system, LLMHolder
from app.models.schemas import QuestionRequest
from app.config.llm2 import get_current_llm

router = APIRouter()
processor = MarkdownProcessor()
rag_chain = None
rag_vector_store = None
llm_holder = None
is_streaming = False

@router.post("/initialize")
async def initialize_rag():
    global rag_chain, rag_vector_store, llm_holder
    try:
        # Pastikan untuk await pemanggilan get_current_llm() agar instance yang dihasilkan bukan coroutine
        llm = await get_current_llm()
        llm_holder = LLMHolder(llm)
        
        # Proses dokumen PDF
        docs = processor.process_markdowns()  
        vector_store_initializer = VectorStoreInitializer()
        rag_vector_store = vector_store_initializer.initialize_vector_store()
        if rag_vector_store.collection.count_documents({}) == 0:
            rag_vector_store.add_documents(docs)

        # Buat RAG chain dengan vector_store dan llm_holder
        rag_chain = create_rag_chain(vector_store=rag_vector_store, llm_holder=llm_holder, streaming=is_streaming)
        return {"message": "RAG system initialized"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Initialization failed: {str(e)}")

@router.post("/ask")
async def ask_question(request: QuestionRequest):
    if not rag_chain:
        raise HTTPException(status_code=400, detail="RAG not initialized")
    try:
        if is_streaming:
            async def generate_stream():
                async for message, metadata in rag_chain.astream(
                    {"question": request.question},
                    stream_mode="messages",
                ):
                    if metadata["langgraph_node"] == "generate":
                        yield message.content.encode("utf-8")  # Encode ke bytes untuk streaming
            return StreamingResponse(generate_stream(), media_type="text/event-stream")
        else:
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
            "app/testsets/testsets.json"
        )
        return evaluation_result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")