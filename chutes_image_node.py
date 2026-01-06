import numpy as np
import requests
import torch
import io
import json
from PIL import Image

class ChutesImageGeneration:
    """Generate images using Chutes.ai Image API."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"multiline": False, "default": ""}),
                "model": (["NovaFurryXL", "iLustMix", "Qwen-Image-2512"], {"default": "NovaFurryXL"}),
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