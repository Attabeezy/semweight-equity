"""
Tests for adaptive sampling
"""

import pytest
import numpy as np
from src.adaptive import AdaptiveSampler


class TestAdaptiveSampler:
    """Test suite for AdaptiveSampler"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.sampler = AdaptiveSampler(sampling_strategy="proportional")
        
    def test_initialization(self):
        """Test sampler initialization"""
        assert self.sampler.sampling_strategy == "proportional"
        assert self.sampler.temperature == 1.0
        assert self.sampler.min_weight == 0.1
        
    def test_compute_sampling_probabilities(self):
        """Test probability computation"""
        weights = np.array([0.5, 0.3, 0.2])
        probs = self.sampler.compute_sampling_probabilities(weights)
        
        assert isinstance(probs, np.ndarray)
        assert len(probs) == len(weights)
        assert np.isclose(probs.sum(), 1.0)
        assert np.all(probs >= 0)
        
    def test_sample_batch(self):
        """Test batch sampling"""
        data = [{"id": i} for i in range(10)]
        weights = np.random.rand(10)
        
        batch = self.sampler.sample_batch(data, weights, batch_size=5)
        
        assert len(batch) == 5
        assert all(isinstance(item, dict) for item in batch)
        
    def test_different_strategies(self):
        """Test different sampling strategies"""
        weights = np.array([0.5, 0.3, 0.2])
        
        for strategy in ["proportional", "sqrt", "log"]:
            sampler = AdaptiveSampler(sampling_strategy=strategy)
            probs = sampler.compute_sampling_probabilities(weights)
            assert np.isclose(probs.sum(), 1.0)

    def test_non_positive_temperature_raises_error(self):
        """Test that a non-positive temperature raises a ValueError"""
        with pytest.raises(ValueError):
            AdaptiveSampler(temperature=0)

        with pytest.raises(ValueError):
            AdaptiveSampler(temperature=-1.0)
