"""
TyDiQA dataset loader
"""

from pathlib import Path
from typing import Dict, List, Optional
import json


def load_tydiqa_split(
    data_dir: Path,
    split: str = "train",
    languages: Optional[List[str]] = None
) -> List[Dict]:
    """
    Load TyDiQA dataset split.
    
    Args:
        data_dir: Directory containing TyDiQA data
        split: Split to load ('train', 'dev', or 'test')
        languages: List of language codes to filter (if None, load all)
        
    Returns:
        List of QA examples
    """
    # TODO: Implement TyDiQA loading
    # Expected format: JSON lines with question, context, answers, language
    raise NotImplementedError


def format_qa_prompt(example: Dict, include_context: bool = True) -> str:
    """
    Format a QA example as a prompt.
    
    Args:
        example: QA example dictionary
        include_context: Whether to include context in prompt
        
    Returns:
        Formatted prompt string
    """
    if include_context:
        return f"Context: {example['context']}\n\nQuestion: {example['question']}\n\nAnswer:"
    else:
        return f"Question: {example['question']}\n\nAnswer:"
