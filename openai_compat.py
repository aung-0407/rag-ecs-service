from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Literal
import logging
import time
from datetime import datetime

from rag_chain import RAGChain

logger = logging.getLogger(__name__)

# Initialize FastAPI app
compat_app = FastAPI(
    title="OpenAI-Compatible RAG API",
    description="OpenAI-compatible interface for RAG queries",
    version="1.0.0"
)

# Add CORS
compat_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG chain
rag_chain_instance = None


# OpenAI-compatible models
class Model(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int
    owned_by: str = "openai"


class ModelList(BaseModel):
    object: Literal["list"] = "list"
    data: List[Model]


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "gpt-4o-mini"
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str = "stop"


class ChatCompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: ChatCompletionUsage


@compat_app.on_event("startup")
async def startup():
    """Initialize RAG chain"""
    global rag_chain_instance
    try:
        logger.info("Initializing RAG chain for OpenAI compatibility layer...")
        rag_chain_instance = RAGChain()
        logger.info("RAG chain initialized")
    except Exception as e:
        logger.error(f"Failed to initialize RAG chain: {e}")
        raise


@compat_app.get("/v1/models", response_model=ModelList)
async def list_models():
    """
    List available models (OpenAI-compatible)
    Required by Open WebUI to discover available models
    """
    return ModelList(
        data=[
            Model(
                id="gpt-4o-mini",
                created=int(time.time()),
                owned_by="openai"
            ),
            Model(
                id="gpt-4o",
                created=int(time.time()),
                owned_by="openai"
            ),
            Model(
                id="gpt-4-turbo",
                created=int(time.time()),
                owned_by="openai"
            ),
            Model(
                id="gpt-3.5-turbo",
                created=int(time.time()),
                owned_by="openai"
            )
        ]
    )


@compat_app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest):
    """
    Chat completions endpoint (OpenAI-compatible)
    
    This endpoint:
    1. Extracts the user's question from messages
    2. Queries the RAG system with specified model
    3. Returns response in OpenAI format
    """
    try:
        if not rag_chain_instance:
            raise HTTPException(status_code=503, detail="RAG system not initialized")
        
        # Extract user query from messages
        user_messages = [msg for msg in request.messages if msg.role == "user"]
        if not user_messages:
            raise HTTPException(status_code=400, detail="No user message found")
        
        query = user_messages[-1].content 
        
        logger.info(f"OpenAI-compat endpoint received query: {query}")
        logger.info(f"Using model: {request.model}")
        
        # Execute RAG query with specified model
        result = rag_chain_instance.query(
            query=query,
            top_k=5,
            search_type="hybrid",
            llm_model=request.model 
        )
        
        answer = result["answer"]
        sources = result["sources"]
        
        # Format answer with sources
        if sources:
            source_info = "\n\n---\n**Sources:**\n"
            for i, source in enumerate(sources[:3], 1): 
                filename = source.get("metadata", {}).get("filename", "Unknown")
                source_info += f"{i}. {filename}\n"
            
            full_answer = answer + source_info
        else:
            full_answer = answer
        
        # Create OpenAI-compatible response
        response = ChatCompletionResponse(
            id=f"chatcmpl-{int(time.time())}",
            created=int(time.time()),
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(
                        role="assistant",
                        content=full_answer
                    ),
                    finish_reason="stop"
                )
            ],
            usage=ChatCompletionUsage(
                prompt_tokens=len(query.split()),
                completion_tokens=len(full_answer.split()),
                total_tokens=len(query.split()) + len(full_answer.split())
            )
        )
        
        logger.info("OpenAI-compat response generated successfully")
        return response
        
    except Exception as e:
        logger.error(f"Chat completion failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@compat_app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "OpenAI-Compatible RAG API",
        "version": "1.0.0",
        "endpoints": {
            "models": "/v1/models",
            "chat": "/v1/chat/completions"
        }
    }


@compat_app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        opensearch_ok = rag_chain_instance.test_opensearch_connection() if rag_chain_instance else False
        openai_ok = bool(rag_chain_instance.openai_api_key) if rag_chain_instance else False
        
        return {
            "status": "healthy" if (opensearch_ok and openai_ok) else "degraded",
            "opensearch_connected": opensearch_ok,
            "openai_configured": openai_ok,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "openai_compat:compat_app",
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )