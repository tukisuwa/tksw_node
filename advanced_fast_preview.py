import torch
import torchvision.transforms as T
import io
import base64
from comfy.utils import common_upscale
from PIL import Image

to_pil_image = T.ToPILImage()

# Check supported image formats
def get_supported_formats():
    """Check which image formats are supported by the current PIL installation"""
    supported = ["JPEG", "PNG"]  # These are always supported

    # Check for WEBP support
    try:
        with io.BytesIO() as test_buffer:
            test_img = Image.new('RGB', (1, 1))
            test_img.save(test_buffer, format='WEBP')
        supported.append("WEBP")
    except (OSError, KeyError):
        pass

    # Check for AVIF support
    try:
        with io.BytesIO() as test_buffer:
            test_img = Image.new('RGB', (1, 1))
            test_img.save(test_buffer, format='AVIF')
        supported.append("AVIF")
    except (OSError, KeyError):
        pass

    return supported

SUPPORTED_FORMATS = get_supported_formats()

# Upscale methods supported by common_upscale
UPSCALE_METHODS = ["nearest", "bilinear", "bicubic", "area", "lanczos"]

class PreviewBase:
    """Base class for preview nodes with shared logic"""

    def process_image(self, image, resize_mode, width, height, format, quality, upscale_method="bilinear"):
        # Original image dimensions
        _b, orig_h, orig_w, _c = image.shape

        target_w, target_h = width, height

        if resize_mode == "auto":
            aspect_ratio = orig_w / orig_h
            if (width / height) > aspect_ratio:
                target_h = height
                target_w = int(height * aspect_ratio)
            else:
                target_w = width
                target_h = int(width / aspect_ratio)

        # Resize the image
        resized_image_tensor = image.permute(0, 3, 1, 2) # B, C, H, W
        resized_image_tensor = common_upscale(resized_image_tensor, target_w, target_h, upscale_method, "center")

        # Convert back to (B, H, W, C) for IMAGE output
        output_tensor = resized_image_tensor.permute(0, 2, 3, 1)

        # Convert to PIL image for preview encoding
        # common_upscale returns (B, C, H, W), so we take the first image
        pil_image = to_pil_image(resized_image_tensor[0])

        with io.BytesIO() as buffered:
            pil_image.save(buffered, format=format, quality=quality)
            img_bytes = buffered.getvalue()

        img_base64 = base64.b64encode(img_bytes).decode('utf-8')

        return (output_tensor, img_base64)


class SimpleFastPreview(PreviewBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "preview"
    CATEGORY = "tksw_node"
    OUTPUT_NODE = True
    DESCRIPTION = "Simple fast preview with automatic settings and image output."

    def preview(self, image):
        # Fixed settings optimized for preview
        output_tensor, img_base64 = self.process_image(
            image=image,
            resize_mode="auto",
            width=512,
            height=512,
            format="JPEG",
            quality=95,
            upscale_method="bilinear"
        )

        return {
            "ui": {"bg_image": [img_base64]},
            "result": (output_tensor,)
        }


class AdvancedFastPreview(PreviewBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", ),
                "resize_mode": (["fixed", "auto"], {"default": "auto"}),
                "width": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 8}),
                "height": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 8}),
                "upscale_method": (UPSCALE_METHODS, {"default": "bilinear"}),
                "format": (SUPPORTED_FORMATS, {"default": "JPEG"}),
                "quality" : ("INT", {"default": 95, "min": 1, "max": 100, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "preview"
    CATEGORY = "tksw_node"
    OUTPUT_NODE = True
    DESCRIPTION = "Advanced preview with resizing capabilities and image output. Can be used as a resize/format conversion node."

    def preview(self, image, resize_mode, width, height, upscale_method, format, quality):
        output_tensor, img_base64 = self.process_image(
            image=image,
            resize_mode=resize_mode,
            width=width,
            height=height,
            format=format,
            quality=quality,
            upscale_method=upscale_method
        )

        return {
            "ui": {"bg_image": [img_base64]},
            "result": (output_tensor,)
        }


NODE_CLASS_MAPPINGS = {
    "SimpleFastPreview": SimpleFastPreview,
    "AdvancedFastPreview": AdvancedFastPreview
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SimpleFastPreview": "Simple Fast Preview",
    "AdvancedFastPreview": "Advanced Fast Preview"
}