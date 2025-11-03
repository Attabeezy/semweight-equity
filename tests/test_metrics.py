"""
Tests for evaluation metrics
"""

import pytest
from src.metrics import (
    compute_exact_match,
    compute_f1_score,
    evaluate_performance
)


class TestMetrics:
    """Test suite for evaluation metrics"""
    
    def test_exact_match(self):
        """Test exact match computation"""
        predictions = ["cat", "dog", "bird"]
        references = ["cat", "dog", "fish"]
        
        em = compute_exact_match(predictions, references)
        assert em == 2/3

    def test_exact_match_empty(self):
        """Test exact match with empty lists"""
        predictions = []
        references = []
        em = compute_exact_match(predictions, references)
        assert em == 0.0
        
    def test_f1_score(self):
        """Test F1 score computation"""
        predictions = ["the cat sat", "dog runs"]
        references = ["the cat sat", "cat runs"]
        
        f1 = compute_f1_score(predictions, references)
        assert 0 <= f1 <= 1
        
    def test_evaluate_performance(self):
        """Test overall performance evaluation"""
        predictions = ["answer1", "answer2"]
        references = ["answer1", "answer3"]
        
        results = evaluate_performance(
            predictions,
            references,
            metrics=["exact_match", "f1"]
        )
        
        assert "exact_match" in results
        assert "f1" in results
        assert 0 <= results["exact_match"] <= 1
        assert 0 <= results["f1"] <= 1
        
    def test_per_language_metrics(self):
        """Test per-language metric computation"""
        predictions = ["answer1", "answer2"]
        references = ["answer1", "answer3"]
        languages = ["en", "fr"]
        
        results = evaluate_performance(
            predictions,
            references,
            languages=languages,
            metrics=["exact_match"]
        )
        
        assert "en_exact_match" in results
        assert "fr_exact_match" in results
