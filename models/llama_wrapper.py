"""
Llama 3 8B model wrapper
"""

from typing import List, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .llm_interface import LLMInterface


class Llama3Wrapper(LLMInterface):
    """
    Wrapper for Llama 3 8B model.
    """
    
    def __init__(
        self,
        model_name: str = "meta-llama/Meta-Llama-3-8B",
        device: Optional[str] = None,
        load_in_8bit: bool = False
    ):
        """
        Initialize Llama 3 model.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to load model on
            load_in_8bit: Whether to use 8-bit quantization
        """
        super().__init__(model_name)
        
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        model_kwargs = {"device_map": "auto"} if load_in_8bit else {}
        if load_in_8bit:
            model_kwargs["load_in_8bit"] = True
            
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )
        
        if not load_in_8bit:
            self.model = self.model.to(self.device)
        
        self.model.eval()
        
    def generate(
        self,
        prompts: List[str],
        max_length: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        **kwargs
    ) -> List[str]:
        """
        Generate responses using Llama 3.
        
        Args:
            prompts: List of input prompts
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            do_sample: Whether to use sampling
            **kwargs: Additional generation parameters
            
        Returns:
            List of generated responses
        """
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
                **kwargs
            )
        
        # Decode only the generated portion
        responses = []
        for i, output in enumerate(outputs):
            input_len = inputs["input_ids"][i].shape[0]
            generated = output[input_len:]
            response = self.tokenizer.decode(generated, skip_special_tokens=True)
            responses.append(response)
            
        return responses
    
    def get_logprobs(
        self,
        prompts: List[str],
        continuations: List[str]
    ) -> List[float]:
        """
        Compute log probabilities of continuations.
        
        Args:
            prompts: List of input prompts
            continuations: List of continuations
            
        Returns:
            List of log probabilities
        """
        logprobs = []
        
        for prompt, continuation in zip(prompts, continuations):
            full_text = prompt + continuation
            
            inputs = self.tokenizer(full_text, return_tensors="pt").to(self.device)
            prompt_inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                
            # Get logprobs for continuation tokens
            prompt_len = prompt_inputs["input_ids"].shape[1]
            continuation_logits = logits[0, prompt_len-1:-1, :]
            continuation_tokens = inputs["input_ids"][0, prompt_len:]
            
            log_probs = torch.nn.functional.log_softmax(continuation_logits, dim=-1)
            token_logprobs = log_probs[range(len(continuation_tokens)), continuation_tokens]
            
            logprobs.append(token_logprobs.sum().item())
            
        return logprobs
