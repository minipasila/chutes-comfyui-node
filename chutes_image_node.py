import numpy as np
import requests
import torch
import io
import json
import time
from PIL import Image

class ChutesImageGeneration:
    """Generate images using Chutes AI Image API."""

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
                "num_inference_steps": ("INT", {"default": 30, "min": 1, "max": 100, "step": 1}),
                "guidance_scale": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "shift": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 10.0, "step": 0.1, "tooltip": "Shift parameter for Flux-based models"}),
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
        guidance_scale=5.0,
        shift=3.0,
    ):
        url = "https://image.chutes.ai/generate"

        # Chutes API Schema Requirements:
        # Seed must be UInt32 (max 4294967295). ComfyUI provides Int64.
        # We allow -1 for random, otherwise modulo the seed to fit 32-bit.
        
        final_seed = None
        if seed != -1:
            # Ensure seed fits in 32-bit integer (0 to 4,294,967,295)
            final_seed = seed % 4294967295

        payload = {
            "model": model,
            "prompt": prompt,
            "width": width,
            "height": height,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "shift": shift,
        }

        # Only add seed if explicitly set
        if final_seed is not None:
            payload["seed"] = final_seed

        # Only add negative prompt if provided (some models might strictly reject it if empty)
        if negative_prompt and negative_prompt.strip():
            payload["negative_prompt"] = negative_prompt

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        print(f"Sending request to Chutes.ai Image API ({model})...")
        print(f"Params: {width}x{height}, Steps: {num_inference_steps}, CFG: {guidance_scale}, Seed: {final_seed}")
        
        response = None
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=180)
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            # Handle specific API errors
            error_msg = f"HTTP Error {response.status_code}"
            if response.status_code == 503:
                error_msg = f"Model '{model}' is currently unavailable (No active miners). Please try again later or choose a different model."
            elif response.status_code == 400:
                try:
                    detail = response.json().get('detail', response.text)
                    error_msg = f"API Validation Error: {detail} (Check image dimensions or parameter limits)"
                except:
                    error_msg = f"Bad Request: {response.text}"
            else:
                 error_msg = f"Request failed: {response.text}"
            
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
                 raise Exception(f"Failed to decode image. Server returned: {response.text[:200]}...")
        else:
             raise Exception(f"Unexpected response type: {content_type}. Body: {response.text}")

        # Convert PIL to Tensor (Batch, Height, Width, Channels)
        img = img.convert("RGB")
        img_np = np.array(img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).unsqueeze(0)

        return (img_tensor,)