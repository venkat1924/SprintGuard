"""
Similar Story Retrieval using BERT Embeddings
Finds historically similar stories for context and benchmarking.
"""
import numpy as np
from typing import List, Dict
from src.models.story import Story
from src.ml.bert_embedder import BertEmbedder


class SimilarityRetriever:
    """
    Retrieves similar historical stories using embedding similarity.
    
    Strategy:
    - Reuse DistilBERT embeddings (no separate model)
    - Brute-force Dot Product (fast enough for <50k stories)
    - Return top-k with metadata
    """
    
    def __init__(self, historical_stories: List[Story], embedder: BertEmbedder = None):
        """
        Initialize retriever with historical stories.
        
        Args:
            historical_stories: List of past stories
            embedder: BertEmbedder instance (creates new if None)
        """
        self.historical_stories = historical_stories
        self.embedder = embedder if embedder is not None else BertEmbedder(quantize=True)
        
        # Precompute embeddings for all historical stories
        print(f"Precomputing embeddings for {len(historical_stories)} stories...")
        self.story_embeddings = self._precompute_embeddings()
        print(f"✓ Similarity retriever ready")
    
    def _precompute_embeddings(self) -> np.ndarray:
        """
        Precompute embeddings for all historical stories.
        
        Returns:
            (N, 768) embedding matrix
        """
        texts = []
        for story in self.historical_stories:
            # Combine title + description
            text = f"{story.title} {story.description}"
            texts.append(text)
        
        # Batch embedding extraction
        embeddings = self.embedder.embed_batch(texts, normalize=True)
        
        return embeddings
    
    def find_similar(
        self,
        query_text: str,
        k: int = 5,
        min_similarity: float = 0.0
    ) -> List[Dict]:
        """
        Find top-k similar stories.
        
        Args:
            query_text: User story text to match
            k: Number of similar stories to return
            min_similarity: Minimum similarity threshold (0-1)
            
        Returns:
            List of dicts with story metadata and similarity scores
        """
        # Embed query
        query_embedding = self.embedder.embed(query_text, normalize=True)
        
        # Compute similarity scores (Dot Product on normalized vectors = Cosine Similarity)
        similarities = self.story_embeddings @ query_embedding  # (N,)
        
        # Get top-k indices
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        
        # Build results
        results = []
        for idx in top_k_indices:
            similarity = similarities[idx]
            
            # Skip if below threshold
            if similarity < min_similarity:
                continue
            
            story = self.historical_stories[idx]
            
            results.append({
                'story_id': story.id,
                'title': story.title,
                'description': story.description[:200] + '...' if len(story.description) > 200 else story.description,
                'risk_level': getattr(story, 'risk_level', 'Unknown'),
                'story_points': getattr(story, 'story_points', None),
                'similarity_score': float(similarity)
            })
        
        return results
    
    def find_similar_by_embedding(
        self,
        query_embedding: np.ndarray,
        k: int = 5
    ) -> List[Dict]:
        """
        Find similar stories given an embedding (avoids recomputation).
        
        Args:
            query_embedding: Pre-computed embedding (768,)
            k: Number of results
            
        Returns:
            List of similar stories
        """
        # Normalize if needed
        if np.linalg.norm(query_embedding) > 1.01 or np.linalg.norm(query_embedding) < 0.99:
            query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        # Compute similarities
        similarities = self.story_embeddings @ query_embedding
        
        # Get top-k
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        
        # Build results
        results = []
        for idx in top_k_indices:
            story = self.historical_stories[idx]
            results.append({
                'story_id': story.id,
                'title': story.title,
                'similarity_score': float(similarities[idx])
            })
        
        return results
    
    def get_embedding_for_story(self, story_id: int) -> np.ndarray:
        """
        Get precomputed embedding for a story by ID.
        
        Args:
            story_id: Story ID
            
        Returns:
            (768,) embedding or None if not found
        """
        for i, story in enumerate(self.historical_stories):
            if story.id == story_id:
                return self.story_embeddings[i]
        return None

