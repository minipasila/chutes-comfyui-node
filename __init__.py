"""
Chutes AI ComfyUI Custom Nodes
Generate images using Chutes AI Image models via API.
"""

from .chutes_image_node import ChutesImageGeneration

NODE_CLASS_MAPPINGS = {
    "ChutesImageGeneration": ChutesImageGeneration
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ChutesImageGeneration": "Chutes Image Gen (API)"
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]