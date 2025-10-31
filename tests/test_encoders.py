"""
Tests for multilingual encoders
"""

import pytest
import torch
from src.encoders import MultilingualEncoder


class TestMultilingualEncoder:
    """Test suite for MultilingualEncoder"""
    
    @pytest.fixture
    def encoder(self):
        """Create encoder fixture"""
        return MultilingualEncoder(model_name="xlm-roberta-base")
        
    def test_initialization(self, encoder):
        """Test encoder initialization"""
        assert encoder.model_name == "xlm-roberta-base"
        assert encoder.tokenizer is not None
        assert encoder.model is not None
        
    def test_encode(self, encoder):
        """Test text encoding"""
        texts = ["Hello", "World"]
        embeddings = encoder.encode(texts)
        
        assert isinstance(embeddings, torch.Tensor)
        assert embeddings.shape[0] == len(texts)
        assert embeddings.shape[1] > 0  # Embedding dimension
        
    def test_compute_similarity(self, encoder):
        """Test similarity computation"""
        texts1 = ["Hello world"]
        texts2 = ["Hello world", "Goodbye"]
        
        emb1 = encoder.encode(texts1)
        emb2 = encoder.encode(texts2)
        
        similarity = encoder.compute_similarity(emb1, emb2)
        
        assert isinstance(similarity, torch.Tensor)
        assert similarity.shape == (1, 2)
        assert similarity[0, 0] > similarity[0, 1]  # Same text should be more similar
