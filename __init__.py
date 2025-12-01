"""
Chutes.ai ComfyUI Custom Nodes
Generate videos using Chutes.ai Wan models via API.
"""

from .chutes_wan_node import ChutesWanVideoFast

NODE_CLASS_MAPPINGS = {
    "ChutesWanVideoFast": ChutesWanVideoFast
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ChutesWanVideoFast": "Chutes Wan 2.2 Fast (API)"
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

