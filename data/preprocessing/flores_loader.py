"""
FLORES-200 dataset loader
"""

from pathlib import Path
from typing import Dict, List, Optional
import json


def load_flores_split(
    data_dir: Path,
    split: str = "dev",
    languages: Optional[List[str]] = None
) -> Dict[str, List[str]]:
    """
    Load FLORES-200 dataset split.
    
    Args:
        data_dir: Directory containing FLORES-200 data
        split: Split to load ('dev' or 'devtest')
        languages: List of language codes to load (if None, load all)
        
    Returns:
        Dictionary mapping language codes to list of sentences
    """
    # TODO: Implement FLORES-200 loading
    # Expected structure: data_dir/{split}/{lang}.{split}
    raise NotImplementedError


def create_translation_pairs(
    flores_data: Dict[str, List[str]],
    source_lang: str,
    target_langs: List[str]
) -> List[Dict]:
    """
    Create translation pairs from FLORES data.
    
    Args:
        flores_data: Dictionary of FLORES sentences by language
        source_lang: Source language code
        target_langs: List of target language codes
        
    Returns:
        List of translation pair dictionaries
    """
    pairs = []
    
    for i, source_text in enumerate(flores_data[source_lang]):
        for target_lang in target_langs:
            pairs.append({
                "source": source_text,
                "target": flores_data[target_lang][i],
                "source_lang": source_lang,
                "target_lang": target_lang,
                "index": i
            })
    
    return pairs
