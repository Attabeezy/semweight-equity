"""
Tests for core SWELM functionality
"""

import pytest
import numpy as np
from src.core import SWELM


class TestSWELM:
    """Test suite for SWELM class"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.swelm = SWELM(encoder_model="xlm-roberta-base", alpha=0.5, beta=1.0)
        
    def test_initialization(self):
        """Test SWELM initialization"""
        assert self.swelm.encoder_model == "xlm-roberta-base"
        assert self.swelm.alpha == 0.5
        assert self.swelm.beta == 1.0
        
    def test_compute_semantic_weights(self):
        """Test semantic weight computation"""
        texts = ["Hello world", "Bonjour monde"]
        languages = ["en", "fr"]
        
        # TODO: Implement after method is complete
        with pytest.raises(NotImplementedError):
            weights = self.swelm.compute_semantic_weights(texts, languages)
            
    def test_reweight_samples(self):
        """Test sample reweighting"""
        samples = [{"text": "sample1"}, {"text": "sample2"}]
        weights = np.array([0.8, 0.2])
        
        # TODO: Implement after method is complete
        with pytest.raises(NotImplementedError):
            reweighted = self.swelm.reweight_samples(samples, weights)
