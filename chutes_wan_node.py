import numpy as np
import requests
import base64
import io
import os
import folder_paths
from PIL import Image


class ChutesWanVideoFast:
    """Generate video from image using Chutes.ai Wan 2.2 Fast API."""

    # Default Chinese negative prompt from the API
    DEFAULT_NEGATIVE = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True, "default": "A cinematic video..."}),
                "api_key": ("STRING", {"multiline": False, "default": ""}),
            },
            "optional": {
                "negative_prompt": ("STRING", {
                    "multiline": True,
                    "default": ChutesWanVideoFast.DEFAULT_NEGATIVE
                }),
                "resolution": (["480p", "720p"], {"default": "480p"}),
                "frames": ("INT", {"default": 81, "min": 21, "max": 140, "step": 1}),
                "fps": ("INT", {"default": 16, "min": 16, "max": 24, "step": 1}),
                "fast": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647}),
                "guidance_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "guidance_scale_2": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_path",)
    FUNCTION = "generate_video"
    CATEGORY = "Chutes/Wan"

    def tensor_to_base64(self, image):
        """Convert ComfyUI tensor (B,H,W,C) to base64 PNG string."""
        i = 255.0 * image.cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8)[0])
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def generate_video(
        self,
        image,
        prompt,
        api_key,
        negative_prompt=None,
        resolution="480p",
        frames=81,
        fps=16,
        fast=True,
        seed=-1,
        guidance_scale=1.0,
        guidance_scale_2=1.0,
    ):
        url = "https://chutes-wan-2-2-i2v-14b-fast.chutes.ai/generate"

        print("Converting image to Base64...")
        image_b64 = self.tensor_to_base64(image)

        payload = {
            "image": image_b64,
            "prompt": prompt,
            "negative_prompt": negative_prompt if negative_prompt else self.DEFAULT_NEGATIVE,
            "resolution": resolution,
            "frames": frames,
            "fps": fps,
            "fast": fast,
            "guidance_scale": guidance_scale,
            "guidance_scale_2": guidance_scale_2,
        }

        # Only include seed if explicitly set (not -1)
        if seed >= 0:
            payload["seed"] = seed
        else:
            payload["seed"] = None

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        print(f"Sending request to Chutes.ai ({url})...")
        print(f"  resolution={resolution}, frames={frames}, fps={fps}, fast={fast}")
        response = None
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=600)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            error_detail = ""
            if response is not None:
                error_detail = f" Response: {response.text}"
                print(f"API Error Response: {response.text}")
            raise Exception(f"Request failed: {e}{error_detail}")

        # Save video - API returns raw video/mp4 bytes
        output_dir = folder_paths.get_output_directory()
        filename = f"chutes_wan_{np.random.randint(100000)}.mp4"
        file_path = os.path.join(output_dir, filename)

        content_type = response.headers.get("Content-Type", "")
        if "application/json" in content_type:
            data = response.json()
            if "url" in data:
                print(f"Downloading video from {data['url']}...")
                video_resp = requests.get(data["url"], timeout=300)
                video_resp.raise_for_status()
                with open(file_path, "wb") as f:
                    f.write(video_resp.content)
            else:
                raise Exception(f"Unexpected JSON response: {data}")
        else:
            with open(file_path, "wb") as f:
                f.write(response.content)

        print(f"Video saved to: {file_path}")
        return (filename,)
