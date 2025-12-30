import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
import re

logger = logging.getLogger(__name__)


class SimilarityMatcher:
    """
    Advanced AI-powered similarity matching service.
    Uses multiple techniques to find similar prompts:
    1. Semantic embeddings (sentence-transformers)
    2. Keyword extraction and matching
    3. Intent classification
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
        self.similarity_threshold = 0.80  # 80% similarity threshold as requested
        
        # Technical keywords for better matching
        self.tech_keywords = {
            "architecture": ["microservices", "monolith", "distributed", "scalable", "load balancer", 
                           "api gateway", "message queue", "event-driven", "serverless", "kubernetes",
                           "docker", "cloud", "aws", "azure", "gcp", "netflix", "youtube", "amazon",
                           "uber", "spotify", "facebook", "instagram", "twitter", "whatsapp"],
            "database": ["sql", "nosql", "postgresql", "mysql", "mongodb", "redis", "cassandra",
                        "dynamodb", "schema", "index", "query", "partition", "sharding", "replication",
                        "acid", "cap theorem", "normalization", "denormalization"],
            "api": ["rest", "graphql", "grpc", "websocket", "oauth", "jwt", "rate limiting",
                   "pagination", "versioning", "swagger", "openapi", "endpoint", "authentication"],
            "ui": ["design", "color", "typography", "font", "layout", "responsive", "mobile",
                  "accessibility", "ux", "user experience", "component", "dashboard", "landing page"],
            "prompts": ["prompt", "template", "ai", "gpt", "claude", "copilot", "llm", "chain of thought"]
        }
    
    def _load_model(self):
        """Lazy load the sentence transformer model"""
        if self._model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
                logger.info("✓ Embedding model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                raise
    
    def get_embedding(self, text: str) -> List[float]:
        """Generate embedding for a text"""
        self._load_model()
        processed_text = self._preprocess_text(text)
        embedding = self._model.encode(processed_text, convert_to_numpy=True)
        return embedding.tolist()
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for better matching"""
        text = text.lower()
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def _extract_keywords(self, text: str) -> set:
        """Extract important keywords from text"""
        text_lower = text.lower()
        keywords = set()
        
        # Extract technical keywords
        for category, terms in self.tech_keywords.items():
            for term in terms:
                if term in text_lower:
                    keywords.add(term)
        
        # Extract quoted terms
        quoted = re.findall(r'"([^"]*)"', text)
        keywords.update([q.lower() for q in quoted])
        
        # Extract capitalized terms (likely proper nouns/technologies)
        caps = re.findall(r'\b[A-Z][a-zA-Z]+\b', text)
        keywords.update([c.lower() for c in caps])
        
        return keywords
    
    def _keyword_similarity(self, query_keywords: set, stored_keywords: set) -> float:
        """Calculate keyword-based similarity using Jaccard index"""
        if not query_keywords or not stored_keywords:
            return 0.0
        
        intersection = query_keywords & stored_keywords
        union = query_keywords | stored_keywords
        
        if not union:
            return 0.0
        
        return len(intersection) / len(union)
    
    def _extract_intent(self, text: str) -> str:
        """Extract the intent/action from the prompt"""
        text_lower = text.lower()
        
        intents = {
            "design": ["design", "architect", "create", "build", "develop"],
            "explain": ["explain", "how does", "what is", "describe", "tell me about"],
            "compare": ["compare", "difference between", "vs", "versus"],
            "optimize": ["optimize", "improve", "scale", "performance"],
            "implement": ["implement", "code", "write", "create a"],
            "troubleshoot": ["fix", "debug", "issue", "problem", "error"]
        }
        
        for intent, keywords in intents.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return intent
        
        return "general"
    
    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))
    
    def calculate_combined_similarity(
        self,
        query: str,
        stored_prompt: str,
        query_embedding: List[float],
        stored_embedding: List[float]
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate combined similarity using multiple methods:
        - 70% weight: Semantic similarity (embeddings)
        - 20% weight: Keyword similarity
        - 10% weight: Intent matching
        """
        # 1. Semantic similarity
        semantic_sim = self.cosine_similarity(query_embedding, stored_embedding)
        
        # 2. Keyword similarity
        query_keywords = self._extract_keywords(query)
        stored_keywords = self._extract_keywords(stored_prompt)
        keyword_sim = self._keyword_similarity(query_keywords, stored_keywords)
        
        # 3. Intent similarity
        query_intent = self._extract_intent(query)
        stored_intent = self._extract_intent(stored_prompt)
        intent_sim = 1.0 if query_intent == stored_intent else 0.3
        
        # Combined score with weights
        combined_score = (
            0.70 * semantic_sim +
            0.20 * keyword_sim +
            0.10 * intent_sim
        )
        
        breakdown = {
            "semantic": semantic_sim,
            "keyword": keyword_sim,
            "intent": intent_sim,
            "combined": combined_score
        }
        
        return combined_score, breakdown
    
    def find_similar_prompt(
        self,
        query: str,
        existing_data: List[Dict[str, Any]],
        threshold: float = None
    ) -> Optional[Tuple[Dict[str, Any], float, Dict[str, float]]]:
        """
        Find the most similar prompt using AI-powered matching.
        
        Returns:
            Tuple of (matching_entry, similarity_score, score_breakdown) or None
        """
        if not existing_data:
            logger.info("No existing data to match against")
            return None
        
        threshold = threshold or self.similarity_threshold
        logger.info(f"🔍 Searching for similar prompts (threshold: {threshold*100:.0f}%)")
        logger.info(f"   Query: {query[:80]}...")
        
        # Generate embedding for query
        query_embedding = self.get_embedding(query)
        
        best_match = None
        best_score = 0.0
        best_breakdown = {}
        
        for entry in existing_data:
            # Get or generate embedding for stored prompt
            if entry.get("embedding"):
                stored_embedding = entry["embedding"]
            else:
                logger.debug(f"Generating embedding for: {entry['prompt'][:50]}...")
                stored_embedding = self.get_embedding(entry["prompt"])
            
            # Calculate combined similarity
            score, breakdown = self.calculate_combined_similarity(
                query,
                entry["prompt"],
                query_embedding,
                stored_embedding
            )
            
            logger.debug(f"  Score {score*100:.1f}% for: {entry['prompt'][:40]}...")
            
            if score > best_score:
                best_score = score
                best_match = entry
                best_breakdown = breakdown
        
        # Log the best match details
        if best_match:
            logger.info(f"📊 Best match found:")
            logger.info(f"   Prompt: {best_match['prompt'][:60]}...")
            logger.info(f"   Combined Score: {best_score*100:.1f}%")
            logger.info(f"   ├─ Semantic: {best_breakdown.get('semantic', 0)*100:.1f}%")
            logger.info(f"   ├─ Keywords: {best_breakdown.get('keyword', 0)*100:.1f}%")
            logger.info(f"   └─ Intent: {best_breakdown.get('intent', 0)*100:.1f}%")
        
        # Use small tolerance (0.001 = 0.1%) for floating point comparison
        tolerance = 0.001
        if best_score >= (threshold - tolerance):
            logger.info(f"✅ MATCH! Score {best_score*100:.1f}% >= threshold {threshold*100:.0f}%")
            return (best_match, best_score, best_breakdown)
        
        logger.info(f"❌ No match (best: {best_score*100:.2f}% below threshold {threshold*100:.0f}%)")
        return None
    
    def batch_generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts efficiently"""
        self._load_model()
        processed_texts = [self._preprocess_text(t) for t in texts]
        embeddings = self._model.encode(processed_texts, convert_to_numpy=True, show_progress_bar=True)
        return embeddings.tolist()
