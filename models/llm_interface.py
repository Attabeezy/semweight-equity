"""
Generic LLM interface for experiments
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional


class LLMInterface(ABC):
    """
    Abstract interface for language models.
    """
    
    def __init__(self, model_name: str, **kwargs):
        """
        Initialize the LLM.
        
        Args:
            model_name: Name/path of the model
            **kwargs: Additional model-specific arguments
        """
        self.model_name = model_name
        
    @abstractmethod
    def generate(
        self,
        prompts: List[str],
        max_length: int = 512,
        temperature: float = 0.7,
        **kwargs
    ) -> List[str]:
        """
        Generate responses for a batch of prompts.
        
        Args:
            prompts: List of input prompts
            max_length: Maximum generation length
            temperature: Sampling temperature
            **kwargs: Additional generation parameters
            
        Returns:
            List of generated responses
        """
        pass
    
    @abstractmethod
    def get_logprobs(
        self,
        prompts: List[str],
        continuations: List[str]
    ) -> List[float]:
        """
        Compute log probabilities of continuations given prompts.
        
        Args:
            prompts: List of input prompts
            continuations: List of continuations
            
        Returns:
            List of log probabilities
        """
        pass
    
    def batch_generate(
        self,
        prompts: List[str],
        batch_size: int = 8,
        **kwargs
    ) -> List[str]:
        """
        Generate responses in batches.
        
        Args:
            prompts: List of prompts
            batch_size: Batch size for processing
            **kwargs: Generation parameters
            
        Returns:
            List of generated responses
        """
        responses = []
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            batch_responses = self.generate(batch, **kwargs)
            responses.extend(batch_responses)
        return responses
