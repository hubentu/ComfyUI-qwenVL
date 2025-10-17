"""
ComfyUI Custom Nodes for Qwen2.5-VL GGUF Models
Provides nodes for loading and running GGUF quantized Qwen2.5-VL models using llama.cpp
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

