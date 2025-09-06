"""
Virtual Fashion Try-On System

A comprehensive AI-powered virtual fashion try-on system that combines:
- Human body parsing using DeepLab architecture with self-correction
- Text-to-garment generation using Stable Diffusion
- Virtual try-on synthesis with realistic blending

This package provides both training and inference capabilities for all components.
"""

__version__ = "1.0.0"
__author__ = "Virtual Fashion Team"
__email__ = "team@virtualfashion.com"

from src.models.human_parser import HumanParsingNet
from src.models.garment_generator import SimpleGarmentGenerator  
from src.models.virtual_tryon import VirtualTryOnPipeline
from src.utils.config import Config

__all__ = [
    "HumanParsingNet",
    "SimpleGarmentGenerator", 
    "VirtualTryOnPipeline",
    "Config"
]