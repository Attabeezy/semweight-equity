"""
Multilingual encoder utilities
"""

import torch
from typing import List, Optional
from transformers import AutoModel, AutoTokenizer


class MultilingualEncoder:
    """
    Wrapper for multilingual encoders (XLM-R, mBERT, LaBSE, etc.)
    """
    
    def __init__(
        self,
        model_name: str = "xlm-roberta-base",
        device: Optional[str] = None
    ):
        """
        Initialize the multilingual encoder.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to run the model on
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
    def encode(
        self,
        texts: List[str],
        batch_size: int = 32,
        max_length: int = 512
    ) -> torch.Tensor:
        """
        Encode texts into embeddings.
        
        Args:
            texts: List of input texts
            batch_size: Batch size for encoding
            max_length: Maximum sequence length
            
        Returns:
            Tensor of embeddings
        """
        embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                
                inputs = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt"
                ).to(self.device)
                
                outputs = self.model(**inputs)
                # Use [CLS] token embedding or mean pooling
                batch_embeddings = outputs.last_hidden_state[:, 0, :]
                embeddings.append(batch_embeddings.cpu())
                
        return torch.cat(embeddings, dim=0)
    
    def compute_similarity(
        self,
        embeddings1: torch.Tensor,
        embeddings2: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute cosine similarity between embeddings.
        
        Args:
            embeddings1: First set of embeddings
            embeddings2: Second set of embeddings
            
        Returns:
            Similarity scores
        """
        embeddings1 = torch.nn.functional.normalize(embeddings1, p=2, dim=1)
        embeddings2 = torch.nn.functional.normalize(embeddings2, p=2, dim=1)
        return torch.mm(embeddings1, embeddings2.t())
