"""
DistilBERT Embedding Extraction with GPU Support, Quantization and Caching
"""
import torch
import numpy as np
from transformers import DistilBertTokenizer, DistilBertModel
from functools import lru_cache
import hashlib
from typing import Optional

from src.ml.config import config


class BertEmbedder:
    """
    Extract 768-dimensional CLS token embeddings from DistilBERT.
    
    Features:
    - Automatic GPU detection and usage
    - Dynamic INT8 quantization for CPU speedup (when no GPU)
    - LRU caching for repeated stories
    - L2 normalization for similarity retrieval
    """
    
    def __init__(
        self,
        model_name: str = None,
        max_length: int = None,
        quantize: bool = None,
        cache_size: int = None
    ):
        """
        Initialize DistilBERT embedder.
        
        Args:
            model_name: HuggingFace model identifier
            max_length: Maximum sequence length (128 for user stories)
            quantize: Apply dynamic INT8 quantization (CPU only)
            cache_size: LRU cache size for embeddings
        """
        # Use config defaults if not specified
        model_name = model_name or config.bert.model_name
        self.max_length = max_length or config.bert.max_length
        self.cache_size = cache_size or config.bert.cache_size
        quantize = quantize if quantize is not None else config.bert.quantize
        
        # Auto-detect device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        print(f"Loading {model_name}...")
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.model = DistilBertModel.from_pretrained(model_name)
        
        # Set to evaluation mode
        self.model.eval()
        
        # Apply quantization only on CPU (GPU uses native FP16/FP32)
        if quantize and self.device.type == "cpu":
            print("Applying dynamic INT8 quantization...")
            self.model = torch.quantization.quantize_dynamic(
                self.model,
                {torch.nn.Linear},
                dtype=torch.qint8
            )
            print("✓ Model quantized (expect 2-3x speedup)")
        elif self.device.type == "cuda":
            print("✓ GPU detected - skipping quantization (using native CUDA)")
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Initialize cache
        self._init_cache()
        
        device_info = f"{self.device}"
        if self.device.type == "cuda":
            device_info += f" ({torch.cuda.get_device_name(0)})"
        print(f"✓ BertEmbedder ready (device={device_info}, max_length={max_length}, cache_size={cache_size})")
    
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
        
        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Forward pass (no gradient computation)
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Extract CLS token (first token) from last hidden state
        # Move to CPU for numpy conversion
        cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
        
        return cls_embedding
    
    def embed_batch(self, texts: list, normalize: bool = True, batch_size: int = None) -> np.ndarray:
        """
        Extract embeddings for multiple texts (batch processing).
        
        Args:
            texts: List of input texts
            normalize: L2-normalize outputs
            batch_size: Number of texts to process at once (for GPU memory management)
            
        Returns:
            (N, 768) numpy array
        """
        batch_size = batch_size or config.bert.batch_size
        all_embeddings = []
        
        # Process in batches to manage memory
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length
            )
            
            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Forward pass
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Extract CLS tokens for all texts and move to CPU
            batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(batch_embeddings)
        
        # Concatenate all batches
        embeddings = np.vstack(all_embeddings)
        
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
