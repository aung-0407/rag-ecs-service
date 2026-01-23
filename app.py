from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import logging
import os
from datetime import datetime

from rag_chain import RAGChain

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="RAG Document Query API",
    description="Query documents using Retrieval-Augmented Generation",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG Chain
rag_chain_instance = None


# Request/Response Models
class QueryRequest(BaseModel):
    query: str = Field(..., description="The question to ask about the documents")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of documents to retrieve")
    search_type: str = Field(default="hybrid", pattern="^(text|vector|hybrid)$")


class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query")
    top_k: int = Field(default=10, ge=1, le=50, description="Number of results")
    search_type: str = Field(default="hybrid", pattern="^(text|vector|hybrid)$")


class DocumentSource(BaseModel):
    content: str
    metadata: Dict[str, Any]
    score: Optional[float] = None


class QueryResponse(BaseModel):
    answer: str
    sources: List[DocumentSource]
    query: str
    timestamp: str
    search_type: str


class SearchResponse(BaseModel):
    results: List[DocumentSource]
    query: str
    total_results: int
    timestamp: str


class HealthResponse(BaseModel):
    status: str
    opensearch_connected: bool
    openai_configured: bool
    timestamp: str


@app.on_event("startup")
async def startup_event():
    """Initialize RAG chain on startup"""
    global rag_chain_instance
    try:
        logger.info("Initializing RAG chain...")
        rag_chain_instance = RAGChain()
        logger.info("RAG chain initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize RAG chain: {str(e)}")
        raise


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "RAG Document Query API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "query": "/query",
            "search": "/search"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint
    Returns the status of the service and its dependencies
    """
    try:
        opensearch_ok = rag_chain_instance.test_opensearch_connection() if rag_chain_instance else False
        openai_ok = bool(rag_chain_instance.openai_api_key) if rag_chain_instance else False
        
        return HealthResponse(
            status="healthy" if (opensearch_ok and openai_ok) else "degraded",
            opensearch_connected=opensearch_ok,
            openai_configured=openai_ok,
            timestamp=datetime.utcnow().isoformat()
        )
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail=f"Health check failed: {str(e)}")


@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Query documents using RAG
    
    This endpoint:
    1. Searches for relevant documents in OpenSearch
    2. Uses retrieved context to generate an answer with OpenAI
    3. Returns the answer with source documents
    """
    try:
        logger.info(f"Received query: {request.query}")
        
        if not rag_chain_instance:
            raise HTTPException(status_code=503, detail="RAG chain not initialized")
        
        # Execute RAG query
        result = rag_chain_instance.query(
            query=request.query,
            top_k=request.top_k,
            search_type=request.search_type
        )
        
        # Format sources
        sources = [
            DocumentSource(
                content=doc.get("content", ""),
                metadata=doc.get("metadata", {}),
                score=doc.get("score")
            )
            for doc in result["sources"]
        ]
        
        response = QueryResponse(
            answer=result["answer"],
            sources=sources,
            query=request.query,
            timestamp=datetime.utcnow().isoformat(),
            search_type=request.search_type
        )
        
        logger.info(f"Query completed successfully. Found {len(sources)} sources.")
        return response
        
    except Exception as e:
        logger.error(f"Query failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@app.post("/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest):
    """
    Search documents without generating an answer
    
    This endpoint only retrieves relevant documents without calling the LLM
    Useful for finding sources or debugging retrieval
    """
    try:
        logger.info(f"Received search request: {request.query}")
        
        if not rag_chain_instance:
            raise HTTPException(status_code=503, detail="RAG chain not initialized")
        
        # Search documents
        documents = rag_chain_instance.search(
            query=request.query,
            top_k=request.top_k,
            search_type=request.search_type
        )
        
        # Format results
        results = [
            DocumentSource(
                content=doc.get("content", ""),
                metadata=doc.get("metadata", {}),
                score=doc.get("score")
            )
            for doc in documents
        ]
        
        response = SearchResponse(
            results=results,
            query=request.query,
            total_results=len(results),
            timestamp=datetime.utcnow().isoformat()
        )
        
        logger.info(f"Search completed. Found {len(results)} documents.")
        return response
        
    except Exception as e:
        logger.error(f"Search failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )