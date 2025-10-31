"""
SWELM: Semantic Weighting for Equitable Language Modeling
"""

__version__ = "0.1.0"

from .core import SWELM
from .encoders import MultilingualEncoder
from .adaptive import AdaptiveSampler
from .metrics import evaluate_performance

__all__ = [
    "SWELM",
    "MultilingualEncoder",
    "AdaptiveSampler",
    "evaluate_performance",
]
