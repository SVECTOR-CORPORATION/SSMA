"""
SSMA Library
------------

This package provides building blocks for creating transformer-like models using
the Structured State Matrix Architecture (SSMA). Users can construct their own models
by combining SSMA layers, hybrid layers, and training utilities.

Example:
    from ssma.model import SSMA_Model
    model = SSMA_Model(...)
"""

__version__ = "0.1.0"

from .layers import HybridSSMALayer, SSMALayer
from .model import SSMA_Model
