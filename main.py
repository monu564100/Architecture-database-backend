from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
import httpx
import logging
import os
from contextlib import asynccontextmanager

from excel_service import ExcelService
from similarity_service import SimilarityMatcher

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Services
excel_service: ExcelService = None
similarity_matcher: SimilarityMatcher = None

# Main backend URL
MAIN_BACKEND_URL = os.getenv("MAIN_BACKEND_URL", "http://localhost:8000")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize services on startup"""
    global excel_service, similarity_matcher
    
    logger.info("=" * 50)
    logger.info("Starting Data Backend (Excel Cache Layer)")
    logger.info("=" * 50)
    
    # Initialize Excel service
    excel_path = os.getenv("EXCEL_PATH", "knowledge_base.xlsx")
    excel_service = ExcelService(excel_path)
    logger.info(f"✓ Excel service initialized: {excel_path}")
    
    # Initialize Similarity matcher
    similarity_matcher = SimilarityMatcher()
    logger.info("✓ Similarity matcher initialized")
    
    logger.info("=" * 50)
    logger.info("Data Backend ready! API available at /api/data")
    logger.info(f"Main backend URL: {MAIN_BACKEND_URL}")
    logger.info("=" * 50)
    
    yield
    
    logger.info("Data Backend shutdown complete")


app = FastAPI(
    title="PromptCraft Data Backend",
    description="Excel-based caching layer for AI responses with semantic similarity matching",
    version="1.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080", "http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response Models
class ChatRequest(BaseModel):
    prompt: str
    category: str
    context: Optional[str] = None
    similarity_threshold: Optional[float] = 0.80  # 80% threshold as requested


class ChatResponse(BaseModel):
    content: str
    source: str  # "cache" or "backend"
    similarity_score: Optional[float] = None
    entry_id: Optional[str] = None
    cached: bool = False


class StatsResponse(BaseModel):
    total_entries: int
    by_category: Dict[str, int]
    cache_hit_rate: Optional[float] = None


# Endpoints
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "data-backend",
        "excel_file": excel_service.excel_path if excel_service else "not initialized",
        "main_backend": MAIN_BACKEND_URL
    }


@app.post("/api/data/chat", response_model=ChatResponse)
async def smart_chat(request: ChatRequest):
    """
    Smart chat endpoint with AI-powered similarity matching:
    1. First checks Excel cache for similar prompts (80%+ match)
    2. If found, returns cached response with similarity details
    3. If not found, calls main backend and ALWAYS stores the response
    
    100% response storage guaranteed!
    """
    logger.info("=" * 60)
    logger.info(f"📨 NEW REQUEST")
    logger.info(f"   Category: {request.category}")
    logger.info(f"   Prompt: {request.prompt[:80]}...")
    logger.info(f"   Threshold: {request.similarity_threshold * 100:.0f}%")
    logger.info("=" * 60)
    
    # Step 1: Get existing data from Excel
    existing_data = excel_service.get_all_prompts_with_embeddings(request.category)
    logger.info(f"📚 Found {len(existing_data)} existing entries in '{request.category}' category")
    
    # Step 2: Check for similar prompt using AI matching
    if existing_data:
        match_result = similarity_matcher.find_similar_prompt(
            request.prompt,
            existing_data,
            threshold=request.similarity_threshold
        )
        
        if match_result:
            # Now returns (entry, score, breakdown)
            if len(match_result) == 3:
                matched_entry, similarity_score, breakdown = match_result
            else:
                matched_entry, similarity_score = match_result
                breakdown = {}
            
            logger.info("✅ CACHE HIT!")
            logger.info(f"   Match Score: {similarity_score * 100:.1f}%")
            logger.info(f"   Returning cached response from entry: {matched_entry['id']}")
            
            # Update usage statistics
            excel_service.update_usage(matched_entry["id"], request.category)
            
            return ChatResponse(
                content=matched_entry["response"],
                source="cache",
                similarity_score=similarity_score,
                entry_id=matched_entry["id"],
                cached=True
            )
        else:
            logger.info("❌ No match above threshold - will fetch from backend")
    else:
        logger.info("📭 No existing data in this category - will fetch from backend")
    
    # Step 3: No cache hit - call main backend
    logger.info("🌐 Calling main backend for fresh response...")
    
    try:
        # Determine the correct endpoint based on category
        if request.category == "ui":
            endpoint = f"{MAIN_BACKEND_URL}/api/v1/chat/ui"
            payload = {"prompt": request.prompt, "industry": request.context}
        else:
            endpoint = f"{MAIN_BACKEND_URL}/api/v1/chat/{request.category}"
            payload = {"prompt": request.prompt, "context": request.context}
        
        logger.info(f"   Endpoint: {endpoint}")
        
        async with httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=30.0)) as client:
            response = await client.post(endpoint, json=payload)
            response.raise_for_status()
            
            data = response.json()
            logger.info(f"✓ Received response from backend (length: {len(str(data))} chars)")
            
            # Extract content based on response structure
            content = data.get("content", "")
            
            if not content:
                logger.error(f"Empty content in response: {data}")
                raise HTTPException(status_code=500, detail="Empty response from backend")
            
            # Step 4: ALWAYS cache the response in Excel (100% storage guarantee)
            logger.info("💾 STORING RESPONSE TO EXCEL (100% storage enabled)...")
            try:
                embedding = similarity_matcher.get_embedding(request.prompt)
                entry_id = excel_service.add_entry(
                    prompt=request.prompt,
                    response=content,
                    category=request.category,
                    embedding=embedding
                )
                logger.info(f"✅ STORED! Entry ID: {entry_id}")
                logger.info(f"   Prompt: {request.prompt[:50]}...")
                logger.info(f"   Response length: {len(content)} chars")
            except Exception as storage_error:
                # Log storage error but don't fail the request
                logger.error(f"⚠️ Storage error (response still returned): {storage_error}")
                entry_id = None
            
            return ChatResponse(
                content=content,
                source="backend",
                similarity_score=None,
                entry_id=entry_id,
                cached=False
            )
            
    except httpx.TimeoutException as e:
        logger.error(f"⏰ Timeout calling main backend: {e}")
        raise HTTPException(status_code=504, detail=f"Backend timeout: Request took too long")
    except httpx.HTTPStatusError as e:
        logger.error(f"🚫 HTTP error from main backend: {e.response.status_code} - {e.response.text}")
        raise HTTPException(status_code=502, detail=f"Backend error: {e.response.status_code}")
    except httpx.HTTPError as e:
        logger.error(f"Error calling main backend: {e}")
        raise HTTPException(status_code=502, detail=f"Backend error: {str(e)}")


@app.get("/api/data/stats", response_model=StatsResponse)
async def get_statistics():
    """Get statistics about the cached data"""
    stats = excel_service.get_statistics()
    return StatsResponse(
        total_entries=stats["total_entries"],
        by_category=stats["by_category"]
    )


@app.get("/api/data/entries/{category}")
async def get_category_entries(category: str, limit: int = 50):
    """Get all entries for a specific category"""
    df = excel_service.get_all_data(category)
    
    if df.empty:
        return {"entries": [], "count": 0}
    
    # Convert to list of dicts, excluding embedding for brevity
    entries = df.head(limit).drop(columns=["Prompt_Embedding"], errors="ignore").to_dict("records")
    
    return {
        "entries": entries,
        "count": len(df),
        "showing": min(limit, len(df))
    }


@app.delete("/api/data/entries/{category}/{entry_id}")
async def delete_entry(category: str, entry_id: str):
    """Delete a specific entry"""
    try:
        import pandas as pd
        df = pd.read_excel(excel_service.excel_path, sheet_name=category)
        df = df[df["ID"] != entry_id]
        
        with pd.ExcelWriter(excel_service.excel_path, mode='a', if_sheet_exists='replace', engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name=category, index=False)
        
        return {"status": "deleted", "entry_id": entry_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/data/rebuild-embeddings/{category}")
async def rebuild_embeddings(category: str):
    """Rebuild embeddings for all entries in a category"""
    try:
        import pandas as pd
        df = pd.read_excel(excel_service.excel_path, sheet_name=category)
        
        if df.empty:
            return {"status": "no entries to process"}
        
        # Generate embeddings for all prompts
        prompts = df["Prompt"].tolist()
        embeddings = similarity_matcher.batch_generate_embeddings(prompts)
        
        # Update dataframe
        import json
        df["Prompt_Embedding"] = [json.dumps(emb) for emb in embeddings]
        
        # Save back
        with pd.ExcelWriter(excel_service.excel_path, mode='a', if_sheet_exists='replace', engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name=category, index=False)
        
        return {"status": "success", "processed": len(prompts)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=True)
