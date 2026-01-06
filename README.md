# Chutes.ai ComfyUI Nodes

Custom nodes for ComfyUI that integrate with [Chutes.ai](https://chutes.ai) image generation API.

## Nodes

### Chutes Image Gen (API)

Generate images using hosted models (NovaFurryXL, iLustMix, Qwen-Image-2512) via the Chutes.ai API.

**Required Inputs:**
| Input | Type | Description |
|-------|------|-------------|
| `api_key` | STRING | Your Chutes.ai API key |
| `model` | COMBO | Select the model (e.g., NovaFurryXL) |
| `prompt` | STRING | Description of the image to generate |
| `width` | INT | Image width (576-2048) |
| `height` | INT | Image height (576-2048) |
| `seed` | INT | Generation seed (-1 = random) |

**Optional Inputs:**
| Input | Type | Default | Description |
|-------|------|---------|-------------|
| `negative_prompt` | STRING | "blur..." | What to avoid in generation |
| `num_inference_steps` | INT | 30 | Step count (1-100) |
| `guidance_scale` | FLOAT | 7.5 | CFG Scale |

**Output:**
- `image`: The generated image (IMAGE tensor) usable by Preview Image or Save Image nodes.

## Installation

### Manual Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/YOUR_USERNAME/chutes-comfyui-node.git
cd chutes-comfyui-node
pip install -r requirements.txt