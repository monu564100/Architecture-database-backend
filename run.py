"""
PromptCraft Data Backend Runner
Excel-based caching layer with semantic similarity matching
"""

import uvicorn
import os

if __name__ == "__main__":
    print("=" * 60)
    print("  PromptCraft Data Backend (Excel Cache Layer)")
    print("=" * 60)
    print()
    print("  This backend provides:")
    print("  • Excel-based storage for prompts and responses")
    print("  • Semantic similarity matching for cache hits")
    print("  • Automatic caching of new responses")
    print()
    print("  Endpoints:")
    print("  • POST /api/data/chat - Smart chat with caching")
    print("  • GET  /api/data/stats - Cache statistics")
    print("  • GET  /api/data/entries/{category} - View cached entries")
    print()
    print("  Running on: http://localhost:8001")
    print("  Main backend: http://localhost:8000")
    print("=" * 60)
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )
