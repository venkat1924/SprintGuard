"""
DistilBERT Embedding Extraction with Quantization and Caching
"""
import torch
import numpy as np
from transformers import DistilBertTokenizer, DistilBertModel
from functools import lru_cache
import hashlib
from typing import Optional


class BertEmbedder:
    """
    Extract 768-dimensional CLS token embeddings from DistilBERT.
    
    Features:
    - Dynamic INT8 quantization for 2-3x CPU speedup
    - LRU caching for repeated stories
    - L2 normalization for similarity retrieval
    """
    
    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        max_length: int = 128,
        quantize: bool = True,
        cache_size: int = 1000
    ):
        """
        Initialize DistilBERT embedder.
        
        Args:
            model_name: HuggingFace model identifier
            max_length: Maximum sequence length (128 for user stories)
            quantize: Apply dynamic INT8 quantization
            cache_size: LRU cache size for embeddings
        """
        self.max_length = max_length
        self.cache_size = cache_size
        
        print(f"Loading {model_name}...")
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.model = DistilBertModel.from_pretrained(model_name)
        
        # Set to evaluation mode
        self.model.eval()
        
        # Apply quantization for CPU inference speedup
        if quantize:
            print("Applying dynamic INT8 quantization...")
            self.model = torch.quantization.quantize_dynamic(
                self.model,
                {torch.nn.Linear},
                dtype=torch.qint8
            )
            print("✓ Model quantized (expect 2-3x speedup)")
        
        # Initialize cache
        self._init_cache()
        
        print(f"✓ BertEmbedder ready (max_length={max_length}, cache_size={cache_size})")
    
    def _init_cache(self):
        """Initialize LRU cache for embeddings."""
        # Cache decorator with instance-specific size
        self._cached_embed = lru_cache(maxsize=self.cache_size)(self._embed_uncached)
    
    def embed(self, text: str, normalize: bool = True) -> np.ndarray:
        """
        Extract embedding for text with caching.
        
        Args:
            text: Input text (title + description)
            normalize: L2-normalize output (for similarity retrieval)
            
        Returns:
            768-dimensional numpy array
        """
        # Create cache key from text hash
        text_hash = hashlib.md5(text.encode()).hexdigest()
        
        # Use cached version
        embedding = self._cached_embed(text_hash, text)
        
        # L2 normalization
        if normalize:
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
        
        return embedding
    
    def _embed_uncached(self, text_hash: str, text: str) -> np.ndarray:
        """
        Internal method for actual embedding extraction.
        
        Args:
            text_hash: Cache key (not used in computation)
            text: Input text
            
        Returns:
            Raw 768-dimensional embedding
        """
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        )
        
        # Forward pass (no gradient computation)
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Extract CLS token (first token) from last hidden state
        cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
        
        return cls_embedding
    
    def embed_batch(self, texts: list, normalize: bool = True) -> np.ndarray:
        """
        Extract embeddings for multiple texts (batch processing).
        
        Args:
            texts: List of input texts
            normalize: L2-normalize outputs
            
        Returns:
            (N, 768) numpy array
        """
        # Tokenize batch
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        )
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Extract CLS tokens for all texts
        embeddings = outputs.last_hidden_state[:, 0, :].numpy()
        
        # L2 normalization
        if normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = np.divide(embeddings, norms, where=(norms > 0))
        
        return embeddings
    
    def get_cache_info(self):
        """Return cache statistics."""
        return self._cached_embed.cache_info()
    
    def clear_cache(self):
        """Clear the embedding cache."""
        self._cached_embed.cache_clear()

