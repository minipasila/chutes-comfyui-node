import numpy as np
import requests
import torch
import io
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
                "width": ("INT", {"default": 1024, "min": 576, "max": 2048, "step": 64}),
                "height": ("INT", {"default": 1024, "min": 576, "max": 2048, "step": 64}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "negative_prompt": ("STRING", {"multiline": True, "default": "blur, distortion, low quality"}),
                "num_inference_steps": ("INT", {"default": 30, "min": 1, "max": 100, "step": 1}),
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

        payload = {
            "model": model,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
        }

        # Handle Seed
        if seed != -1:
            payload["seed"] = seed

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        print(f"Sending request to Chutes.ai Image API ({model})...")
        
        try:
            # We use stream=False as the schema indicates direct image/png response
            response = requests.post(url, json=payload, headers=headers, timeout=120)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            error_detail = ""
            if response is not None:
                try:
                    error_detail = f" Response: {response.text}"
                except:
                    pass
            raise Exception(f"Request failed: {e}{error_detail}")

        # Check content type to ensure we got an image
        content_type = response.headers.get("Content-Type", "")
        
        img = None
        if "image" in content_type or len(response.content) > 0:
            try:
                img = Image.open(io.BytesIO(response.content))
            except Exception as e:
                 # If direct image load fails, check if it's a streaming/json error response
                 raise Exception(f"Failed to decode image data. Content-Type: {content_type}. Error: {e}")
        else:
             raise Exception(f"Unexpected response type: {content_type}")

        # Convert PIL to Tensor (Batch, Height, Width, Channels)
        img = img.convert("RGB")
        img_np = np.array(img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).unsqueeze(0)

        return (img_tensor,)