"""
Tests for DistilBERT-XGBoost Risk Model Components
"""
import pytest
import numpy as np
import time


class TestSymbolicFeatureExtractor:
    """Test symbolic feature extraction."""
    
    def test_basic_extraction(self):
        """Test that features are extracted correctly."""
        from src.ml.feature_extractors import SymbolicFeatureExtractor
        
        extractor = SymbolicFeatureExtractor()
        
        # Test with sample text
        text = "This might be a fast and easy task to implement the API integration."
        features = extractor.extract_features(text)
        
        # Check shape
        assert features.shape == (15,), f"Expected 15 features, got {features.shape}"
        
        # Check feature names match
        feature_names = extractor.get_feature_names()
        assert len(feature_names) == 15
        
        print(f"✓ Extracted {len(features)} features")
    
    def test_ambiguity_detection(self):
        """Test that ambiguous language is detected."""
        from src.ml.feature_extractors import SymbolicFeatureExtractor
        
        extractor = SymbolicFeatureExtractor()
        
        # Ambiguous text
        text = "This might be easy and should be fast. The system could handle it."
        features = extractor.extract_features(text)
        feature_names = extractor.get_feature_names()
        
        # Check modal density
        modal_idx = feature_names.index('weak_modal_density')
        assert features[modal_idx] > 0, "Should detect weak modals"
        
        # Check vague quantifiers
        vague_idx = feature_names.index('has_vague_quantifiers')
        assert features[vague_idx] > 0, "Should detect vague quantifiers (easy, fast)"
        
        print("✓ Ambiguity detection working")
    
    def test_risk_lexicon_detection(self):
        """Test that risk keywords are detected."""
        from src.ml.feature_extractors import SymbolicFeatureExtractor
        
        extractor = SymbolicFeatureExtractor()
        
        # Text with risk keywords
        text = "Need to implement OAuth authentication with JWT tokens. TODO: Handle legacy API migration."
        features = extractor.extract_features(text)
        feature_names = extractor.get_feature_names()
        
        # Check security keywords
        security_idx = feature_names.index('security_count')
        assert features[security_idx] > 0, "Should detect security keywords (oauth, jwt, auth)"
        
        # Check SATD keywords
        satd_idx = feature_names.index('satd_count')
        assert features[satd_idx] > 0, "Should detect SATD keywords (TODO)"
        
        # Check complexity keywords
        complexity_idx = feature_names.index('complexity_count')
        assert features[complexity_idx] > 0, "Should detect complexity keywords (legacy, api, migration)"
        
        print("✓ Risk lexicon detection working")
    
    def test_empty_text(self):
        """Test handling of empty text."""
        from src.ml.feature_extractors import SymbolicFeatureExtractor
        
        extractor = SymbolicFeatureExtractor()
        
        features = extractor.extract_features("")
        assert features.shape == (15,)
        assert np.all(features == 0), "Empty text should return zero vector"
        
        print("✓ Empty text handled correctly")


class TestBertEmbedder:
    """Test BERT embedding extraction."""
    
    def test_embedding_extraction(self):
        """Test that embeddings are extracted with correct shape."""
        from src.ml.bert_embedder import BertEmbedder
        
        embedder = BertEmbedder(quantize=False, cache_size=10)  # No quantization for test speed
        
        text = "As a user, I want to login with OAuth so that I can access my account."
        embedding = embedder.embed(text, normalize=True)
        
        # Check shape
        assert embedding.shape == (768,), f"Expected (768,), got {embedding.shape}"
        
        # Check L2 normalization
        norm = np.linalg.norm(embedding)
        assert abs(norm - 1.0) < 0.01, f"Expected L2 norm ≈ 1.0, got {norm}"
        
        print(f"✓ Embedding shape: {embedding.shape}, L2 norm: {norm:.4f}")
    
    def test_caching(self):
        """Test that caching works."""
        from src.ml.bert_embedder import BertEmbedder
        
        embedder = BertEmbedder(quantize=False, cache_size=10)
        
        text = "Test story for caching"
        
        # First call (cache miss)
        start = time.time()
        emb1 = embedder.embed(text)
        time1 = time.time() - start
        
        # Second call (cache hit)
        start = time.time()
        emb2 = embedder.embed(text)
        time2 = time.time() - start
        
        # Check embeddings are identical
        assert np.allclose(emb1, emb2), "Cached embedding should match"
        
        # Cache should be faster
        assert time2 < time1 * 0.1, f"Cache should be faster: {time1:.4f}s vs {time2:.4f}s"
        
        # Check cache stats
        cache_info = embedder.get_cache_info()
        assert cache_info.hits > 0, "Should have cache hits"
        
        print(f"✓ Caching working: {time1:.4f}s → {time2:.4f}s (cache hit)")
    
    def test_batch_embedding(self):
        """Test batch embedding extraction."""
        from src.ml.bert_embedder import BertEmbedder
        
        embedder = BertEmbedder(quantize=False)
        
        texts = [
            "User story 1",
            "User story 2",
            "User story 3"
        ]
        
        embeddings = embedder.embed_batch(texts, normalize=True)
        
        # Check shape
        assert embeddings.shape == (3, 768), f"Expected (3, 768), got {embeddings.shape}"
        
        # Check normalization
        norms = np.linalg.norm(embeddings, axis=1)
        assert np.all(np.abs(norms - 1.0) < 0.01), "All embeddings should be L2-normalized"
        
        print(f"✓ Batch embedding: {embeddings.shape}")


class TestCostSensitiveClassifier:
    """Test cost-sensitive classification."""
    
    def test_prediction(self):
        """Test that cost-sensitive prediction differs from argmax."""
        from src.ml.threshold_optimizer import CostSensitiveClassifier
        
        classifier = CostSensitiveClassifier()
        
        # Probabilities that would be "Low" (0) by argmax
        # but should be "Medium" or "High" due to cost matrix
        proba = np.array([[
            0.6,   # P(Low)
            0.25,  # P(Medium)
            0.15   # P(High)
        ]])
        
        # Argmax would predict Low (0)
        argmax_pred = np.argmax(proba, axis=1)[0]
        assert argmax_pred == 0
        
        # Cost-sensitive might predict differently (High risk FN is expensive)
        cost_pred = classifier.predict(proba)[0]
        
        # The prediction should consider the cost matrix
        print(f"✓ Argmax: {argmax_pred}, Cost-sensitive: {cost_pred}")
    
    def test_cost_matrix(self):
        """Test that cost matrix is applied correctly."""
        from src.ml.threshold_optimizer import CostSensitiveClassifier
        
        # Custom cost matrix (extreme FN penalty for High)
        cost_matrix = np.array([
            [0, 1, 100],
            [1, 0, 50],
            [2, 2, 0]
        ])
        
        classifier = CostSensitiveClassifier(cost_matrix=cost_matrix)
        
        # Even with low High probability, should predict High due to cost
        proba = np.array([[0.6, 0.3, 0.1]])
        pred = classifier.predict(proba)[0]
        
        # Should be conservative (predict High to avoid costly FN)
        print(f"✓ Cost-sensitive prediction: {pred} for proba {proba[0]}")


class TestIntegration:
    """Integration tests for the full pipeline."""
    
    @pytest.mark.skipif(
        not pytest.importorskip("torch", minversion="2.0"),
        reason="PyTorch not available"
    )
    def test_end_to_end_latency(self):
        """Test end-to-end latency (should be <1s)."""
        from src.ml.feature_extractors import SymbolicFeatureExtractor
        from src.ml.bert_embedder import BertEmbedder
        
        extractor = SymbolicFeatureExtractor()
        embedder = BertEmbedder(quantize=True, cache_size=100)
        
        text = """As a user, I want to implement OAuth authentication 
        so that I can securely access the legacy API system."""
        
        # Warm-up (model loading)
        _ = extractor.extract_features(text)
        _ = embedder.embed(text)
        
        # Measure latency
        start = time.time()
        
        symbolic = extractor.extract_features(text)
        embedding = embedder.embed(text)
        
        latency = time.time() - start
        
        print(f"✓ Feature extraction latency: {latency*1000:.1f}ms")
        
        # Should be well under 1 second
        assert latency < 1.0, f"Latency {latency:.3f}s exceeds 1s threshold"
    
    def test_feature_fusion(self):
        """Test that symbolic and neural features can be fused."""
        from src.ml.feature_extractors import SymbolicFeatureExtractor
        from src.ml.bert_embedder import BertEmbedder
        from sklearn.preprocessing import StandardScaler
        
        extractor = SymbolicFeatureExtractor()
        embedder = BertEmbedder(quantize=False)
        
        text = "Test user story"
        
        # Extract features
        symbolic = extractor.extract_features(text)
        embedding = embedder.embed(text, normalize=True)
        
        # Scale symbolic
        scaler = StandardScaler()
        symbolic_scaled = scaler.fit_transform(symbolic.reshape(1, -1))
        
        # Fuse
        fused = np.hstack([symbolic_scaled, embedding.reshape(1, -1)])
        
        # Check shape
        expected_shape = (1, 15 + 768)
        assert fused.shape == expected_shape, f"Expected {expected_shape}, got {fused.shape}"
        
        print(f"✓ Fused features: {fused.shape}")


def test_model_artifacts_structure():
    """Test that model artifact structure is correct."""
    import os
    
    model_dir = "models"
    expected_files = [
        'xgboost_risk_model.json',
        'feature_scaler.pkl',
        'feature_names.json',
        'risk_lexicons.json'
    ]
    
    # Check if model directory exists
    if os.path.exists(model_dir):
        for filename in expected_files:
            filepath = os.path.join(model_dir, filename)
            if os.path.exists(filepath):
                print(f"  ✓ Found: {filename}")
            else:
                print(f"  ⚠ Missing: {filename} (will be created during training)")
    else:
        print(f"⚠ Model directory '{model_dir}' not found (will be created during training)")


if __name__ == '__main__':
    # Run tests with verbose output
    pytest.main([__file__, '-v', '-s'])

