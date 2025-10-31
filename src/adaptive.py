"""
Adaptive sampling strategies for SWELM
"""

import numpy as np
from typing import List, Dict, Optional, Callable


class AdaptiveSampler:
    """
    Implements adaptive sampling strategies based on semantic weights.
    """
    
    def __init__(
        self,
        sampling_strategy: str = "proportional",
        temperature: float = 1.0,
        min_weight: float = 0.1
    ):
        """
        Initialize the adaptive sampler.
        
        Args:
            sampling_strategy: Strategy for sampling ('proportional', 'sqrt', 'log')
            temperature: Temperature for softmax sampling
            min_weight: Minimum weight threshold
        """
        self.sampling_strategy = sampling_strategy
        self.temperature = temperature
        self.min_weight = min_weight
        
    def compute_sampling_probabilities(
        self,
        weights: np.ndarray
    ) -> np.ndarray:
        """
        Convert semantic weights to sampling probabilities.
        
        Args:
            weights: Semantic weights
            
        Returns:
            Sampling probabilities
        """
        weights = np.maximum(weights, self.min_weight)
        
        if self.sampling_strategy == "proportional":
            probs = weights / weights.sum()
        elif self.sampling_strategy == "sqrt":
            sqrt_weights = np.sqrt(weights)
            probs = sqrt_weights / sqrt_weights.sum()
        elif self.sampling_strategy == "log":
            log_weights = np.log(weights + 1)
            probs = log_weights / log_weights.sum()
        else:
            raise ValueError(f"Unknown sampling strategy: {self.sampling_strategy}")
            
        # Apply temperature
        if self.temperature != 1.0:
            probs = np.power(probs, 1.0 / self.temperature)
            probs = probs / probs.sum()
            
        return probs
    
    def sample_batch(
        self,
        data: List[Dict],
        weights: np.ndarray,
        batch_size: int,
        replace: bool = False
    ) -> List[Dict]:
        """
        Sample a batch according to semantic weights.
        
        Args:
            data: List of data samples
            weights: Semantic weights for each sample
            batch_size: Number of samples to draw
            replace: Whether to sample with replacement
            
        Returns:
            Sampled batch
        """
        probs = self.compute_sampling_probabilities(weights)
        indices = np.random.choice(
            len(data),
            size=batch_size,
            replace=replace,
            p=probs
        )
        return [data[i] for i in indices]
