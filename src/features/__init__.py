# Features module
"""
Feature engineering for match prediction.

Modules:
- engineering: AdvancedFeatureEngineering class for creating 33+ features
- edge_cases: EdgeCaseHandler for detecting and handling problematic inputs
"""

from .engineering import AdvancedFeatureEngineering
from .edge_cases import EdgeCaseHandler

__all__ = ["AdvancedFeatureEngineering", "EdgeCaseHandler"]
