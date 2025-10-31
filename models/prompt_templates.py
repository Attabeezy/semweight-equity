"""
Task-specific prompt templates
"""

from typing import Dict, List


class PromptTemplate:
    """Base class for prompt templates"""
    
    def format(self, example: Dict) -> str:
        """Format an example as a prompt"""
        raise NotImplementedError


class TranslationPrompt(PromptTemplate):
    """Prompt template for translation tasks"""
    
    def __init__(self, source_lang: str, target_lang: str):
        self.source_lang = source_lang
        self.target_lang = target_lang
        
    def format(self, example: Dict) -> str:
        """
        Format translation prompt.
        
        Args:
            example: Dictionary with 'source' text
            
        Returns:
            Formatted prompt
        """
        return f"""Translate the following text from {self.source_lang} to {self.target_lang}:

{self.source_lang}: {example['source']}
{self.target_lang}:"""


class QAPrompt(PromptTemplate):
    """Prompt template for question answering"""
    
    def __init__(self, include_context: bool = True):
        self.include_context = include_context
        
    def format(self, example: Dict) -> str:
        """
        Format QA prompt.
        
        Args:
            example: Dictionary with 'question' and optionally 'context'
            
        Returns:
            Formatted prompt
        """
        if self.include_context and 'context' in example:
            return f"""Answer the question based on the context below.

Context: {example['context']}

Question: {example['question']}

Answer:"""
        else:
            return f"""Question: {example['question']}

Answer:"""


class ReasoningPrompt(PromptTemplate):
    """Prompt template for reasoning tasks"""
    
    def __init__(self, use_chain_of_thought: bool = True):
        self.use_chain_of_thought = use_chain_of_thought
        
    def format(self, example: Dict) -> str:
        """
        Format reasoning prompt.
        
        Args:
            example: Dictionary with 'question'
            
        Returns:
            Formatted prompt
        """
        if self.use_chain_of_thought:
            return f"""Answer the following yes/no question. Think step by step before providing your final answer.

Question: {example['question']}

Let's think step by step:"""
        else:
            return f"""Answer the following yes/no question.

Question: {example['question']}

Answer (yes or no):"""


def get_few_shot_examples(task: str, language: str) -> List[Dict]:
    """
    Get few-shot examples for a task and language.
    
    Args:
        task: Task name ('translation', 'qa', 'reasoning')
        language: Language code
        
    Returns:
        List of example dictionaries
    """
    # TODO: Implement few-shot example retrieval
    return []
