# Chutes AI ComfyUI Nodes

Custom nodes for ComfyUI that integrate with [Chutes AI](https://chutes.ai) image generation API.

## Nodes

### Chutes Image Gen (API)

Generate images using hosted image generation models via the Chutes.ai API. The node automatically fetches available models from the Chutes API.

**Required Inputs:**
| Input | Type | Description |
|-------|------|-------------|
| `api_key` | STRING | Your Chutes.ai API key |
| `model` | COMBO | Select the model (dynamically fetched from API) |
| `prompt` | STRING | Description of the image to generate |
| `width` | INT | Image width (256-2048) |
| `height` | INT | Image height (256-2048) |
| `seed` | INT | Generation seed (-1 = random) |

**Optional Inputs:**
| Input | Type | Default | Description |
|-------|------|---------|-------------|
| `negative_prompt` | STRING | "blur..." | What to avoid in generation |
| `num_inference_steps` | INT | 30 | Step count (1-75) |
| `guidance_scale` | FLOAT | 7.5 | CFG Scale (0.0-20.0) |

**Output:**
- `image`: The generated image (IMAGE tensor) usable by Preview Image or Save Image nodes.

## Dynamic Model Fetching

The node automatically fetches available image generation models from the Chutes API. There are two ways to enable dynamic model fetching:

### Option 1: Environment Variable (Recommended)

Set the `CHUTES_API_KEY` environment variable before starting ComfyUI:

**Linux/Mac:**
```bash
export CHUTES_API_KEY="your_api_key_here"
python main.py
```

**Windows (PowerShell):**
```powershell
$env:CHUTES_API_KEY="your_api_key_here"
python main.py
```

**Windows (CMD):**
```cmd
set CHUTES_API_KEY=your_api_key_here
python main.py
```

### Option 2: Manual API Key

If no environment variable is set, the node will use a fallback list of models (NovaFurryXL, iLustMix, Qwen-Image-2512). You can still provide your API key in the node input to use any model.

### How It Works

1. **On Node Load**: When ComfyUI loads, the node attempts to fetch available models from the Chutes API using the environment variable
2. **Caching**: Fetched models are cached for 5 minutes to avoid repeated API calls
3. **Fallback**: If the API call fails or no API key is provided, the node uses a hardcoded list of popular models
4. **Model Filtering**: The node automatically identifies image generation models by filtering for chutes with diffusion-related templates, images, or descriptions

### Model-Specific Parameters

Different models may have different parameter requirements:

- **Qwen-Image-2512**: Uses `true_cfg_scale` instead of `guidance_scale`, max steps = 75
- **NovaFurryXL & iLustMix**: Use `guidance_scale`, max steps = 50

The node automatically handles these differences based on the selected model.

## Installation

### Manual Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/minipasila/chutes-comfyui-node.git
cd chutes-comfyui-node
pip install -r requirements.txt