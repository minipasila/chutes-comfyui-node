import numpy as np
import requests
import torch
import io
import json
import os
import time
from PIL import Image

# Module-level cache for fetched models
_MODELS_CACHE = {
    'models': None,
    'timestamp': 0,
    'cache_duration': 300,  # 5 minutes
}

# Hardcoded fallback models
FALLBACK_MODELS = ["NovaFurryXL", "iLustMix", "Qwen-Image-2512"]

def fetch_available_chutes(api_key):
    """
    Fetch available chutes from Chutes API.
    
    Args:
        api_key: Chutes API key for authentication
        
    Returns:
        List of chute dictionaries or None on error
    """
    url = "https://api.chutes.ai/chutes/"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    params = {
        "include_public": True,
        "limit": 100
    }
    
    try:
        response = requests.get(url, headers=headers, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        # Handle different response formats
        if isinstance(data, dict) and 'items' in data:
            return data['items']
        elif isinstance(data, list):
            return data
        else:
            print(f"Unexpected response format from Chutes API: {type(data)}")
            return None
            
    except requests.exceptions.HTTPError as e:
        print(f"HTTP error fetching chutes: {e}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Network error fetching chutes: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error fetching chutes: {e}")
        return None

def safe_to_string(value):
    """
    Safely convert a value to a string for keyword matching.
    
    Args:
        value: Any value from API response (str, dict, list, etc.)
        
    Returns:
        String representation of the value, or empty string if None
    """
    if value is None:
        return ""
    
    if isinstance(value, str):
        return value
    
    if isinstance(value, dict):
        # Try to extract common fields from dict
        # Priority: name > title > description > url > string representation
        for key in ['name', 'title', 'description', 'url']:
            if key in value and isinstance(value[key], str):
                return value[key]
        # Fallback to string representation
        return str(value)
    
    if isinstance(value, (list, tuple)):
        # Join list items with spaces
        return " ".join(str(item) for item in value)
    
    # For any other type, convert to string
    return str(value)

def filter_image_generation_chutes(chutes):
    """
    Filter chutes to identify image generation models.
    
    Args:
        chutes: List of chute dictionaries
        
    Returns:
        List of model names (strings)
    """
    if not chutes:
        return []
    
    image_models = []
    
    for chute in chutes:
        # Extract relevant fields with safe conversion
        chute_name = safe_to_string(chute.get('name', ''))
        template = safe_to_string(chute.get('template', ''))
        image = safe_to_string(chute.get('image', ''))
        tagline = safe_to_string(chute.get('tagline', ''))
        tool_description = safe_to_string(chute.get('tool_description', ''))
        
        # Keywords that indicate image generation
        image_keywords = [
            'diffusion', 'stable', 'flux', 'image', 'generate', 'sd', 
            'midjourney', 'dall', 'imagen', 'novafurry', 'ilustmix', 'qwen'
        ]
        
        # Check for image generation indicators
        is_image_model = False
        
        # Check template
        if template and any(keyword in template.lower() for keyword in image_keywords):
            is_image_model = True
        
        # Check image field
        if image and any(keyword in image.lower() for keyword in image_keywords):
            is_image_model = True
        
        # Check name
        if chute_name and any(keyword in chute_name.lower() for keyword in image_keywords):
            is_image_model = True
        
        # Check tagline and description
        text_content = f"{tagline} {tool_description}".lower()
        if text_content and any(keyword in text_content for keyword in image_keywords):
            is_image_model = True
        
        # If identified as image model, add to list
        if is_image_model and chute_name:
            image_models.append(chute_name)
    
    return image_models

def get_cached_models(api_key):
    """
    Get cached models or fetch if cache is expired.
    
    Args:
        api_key: Chutes API key for authentication
        
    Returns:
        List of model names
    """
    current_time = time.time()
    
    # Check if we have valid cached models
    if (_MODELS_CACHE['models'] is not None and 
        current_time - _MODELS_CACHE['timestamp'] < _MODELS_CACHE['cache_duration']):
        return _MODELS_CACHE['models']
    
    # Try to fetch fresh models
    if api_key and api_key.strip():
        chutes = fetch_available_chutes(api_key)
        if chutes:
            image_models = filter_image_generation_chutes(chutes)
            
            if image_models:
                # Update cache
                _MODELS_CACHE['models'] = image_models
                _MODELS_CACHE['timestamp'] = current_time
                print(f"Successfully fetched {len(image_models)} image generation models from Chutes API")
                return image_models
            else:
                print("No image generation models found in fetched chutes, using fallback")
        else:
            print("Failed to fetch chutes from API, using fallback models")
    else:
        print("No API key provided, using fallback models")
    
    # Return fallback models if fetching failed
    _MODELS_CACHE['models'] = FALLBACK_MODELS
    _MODELS_CACHE['timestamp'] = current_time
    return FALLBACK_MODELS

def get_available_models():
    """
    Get available models, trying environment variable first.
    
    Returns:
        List of model names
    """
    # Try to get API key from environment variable
    api_key = os.environ.get('CHUTES_API_KEY', '')
    return get_cached_models(api_key)

class ChutesImageGeneration:
    """Generate images using Chutes.ai Image API."""

    @classmethod
    def INPUT_TYPES(cls):
        # Get available models (dynamic or fallback)
        models = get_available_models()
        default_model = models[0] if models else "NovaFurryXL"
        
        return {
            "required": {
                "api_key": ("STRING", {"multiline": False, "default": ""}),
                "model": (models, {"default": default_model}),
                "prompt": ("STRING", {"multiline": True, "default": "A beautiful sunset over mountains"}),
                "width": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 64}),
                "height": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 64}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "negative_prompt": ("STRING", {"multiline": True, "default": "blur, distortion, low quality"}),
                "num_inference_steps": ("INT", {"default": 30, "min": 1, "max": 75, "step": 1}),
                "guidance_scale": ("FLOAT", {"default": 7.5, "min": 0.0, "max": 20.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate_image"
    CATEGORY = "Chutes/Image"

    def generate_image(
        self,
        api_key,
        model,
        prompt,
        width,
        height,
        seed,
        negative_prompt="",
        num_inference_steps=30,
        guidance_scale=7.5,
    ):
        url = "https://image.chutes.ai/generate"

        # 1. Handle Seed (ComfyUI int64 -> API uint32)
        final_seed = None
        if seed != -1:
            final_seed = seed % 4294967295

        # 2. Base Payload
        payload = {
            "model": model,
            "prompt": prompt,
            "width": width,
            "height": height,
        }

        # 3. Model-Specific Parameter Handling
        if model == "Qwen-Image-2512":
            # Qwen uses 'true_cfg_scale' instead of 'guidance_scale'
            # Qwen Max Steps = 75
            payload["true_cfg_scale"] = guidance_scale
            payload["num_inference_steps"] = min(num_inference_steps, 75)
            # Qwen usually handles negative_prompt, but some versions might ignore it. Included safely.
            if negative_prompt and negative_prompt.strip():
                payload["negative_prompt"] = negative_prompt
        
        else:
            # NovaFurryXL and iLustMix (SDXL/Flux based)
            # Schema: guidance_scale (1-20), num_inference_steps (1-50)
            
            # Enforce strict step limit (Schema max is 50)
            safe_steps = min(num_inference_steps, 50)
            
            payload["guidance_scale"] = guidance_scale
            payload["num_inference_steps"] = safe_steps
            
            if negative_prompt and negative_prompt.strip():
                payload["negative_prompt"] = negative_prompt

        # Add seed if valid
        if final_seed is not None:
            payload["seed"] = final_seed

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        print(f"Sending request to Chutes.ai ({model})...")
        print(f"Payload keys: {list(payload.keys())}") 

        try:
            response = requests.post(url, json=payload, headers=headers, timeout=180)
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            error_msg = f"HTTP Error {response.status_code}"
            
            # Attempt to parse detailed JSON error
            try:
                error_json = response.json()
                detail = error_json.get('detail', str(error_json))
            except:
                detail = response.text

            if response.status_code == 503:
                error_msg = f"Service Unavailable for '{model}': {detail}. Try again later."
            elif response.status_code == 400:
                error_msg = f"Validation Error (400): {detail}"
            else:
                error_msg = f"API Error {response.status_code}: {detail}"
            
            raise Exception(error_msg) from e
        except requests.exceptions.RequestException as e:
            raise Exception(f"Connection error: {e}")

        # Process Image Response
        content_type = response.headers.get("Content-Type", "")
        img = None
        
        if "image" in content_type or len(response.content) > 0:
            try:
                img = Image.open(io.BytesIO(response.content))
            except Exception as e:
                 raise Exception(f"Failed to decode image. Bytes received: {len(response.content)}. Error: {e}")
        else:
             raise Exception(f"Unexpected response type: {content_type}. Body: {response.text}")

        img = img.convert("RGB")
        img_np = np.array(img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).unsqueeze(0)

        return (img_tensor,)