import os
from PIL import Image
import numpy as np
import torch
import random

class ImageSequenceLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "folder_path": ("STRING", {"default": ""}),
                "reset": ("BOOLEAN", {"default": False}),
                "loop_or_reset": ("BOOLEAN", {"default": False}),
                "reset_on_error": ("BOOLEAN", {"default": False}),
                "exclude_loaded_on_reset": ("BOOLEAN", {"default": False}),
                "output_alpha": ("BOOLEAN", {"default": False}),
                "include_extension": ("BOOLEAN", {"default": False}),
                "use_manual_index": ("BOOLEAN", {"default": False}),
                "start_index": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "manual_index": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "STRING")
    RETURN_NAMES = ("image", "index", "seed", "filename")
    FUNCTION = "run"
    CATEGORY = "image"

    def __init__(self):
        self.current_index = 0
        self.image_files = []
        self.prev_folder_path = ""

    def _load_image_files(self, folder_path):
        self.image_files = sorted([
            f for f in os.listdir(folder_path)
            if os.path.isfile(os.path.join(folder_path, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
        ])

    def _load_image(self, folder_path, index, alpha):
        if 0 <= index < len(self.image_files):
            image_path = os.path.join(folder_path, self.image_files[index])
            filename = self.image_files[index]
            try:
                with Image.open(image_path) as image:
                    if alpha == False:
                        image = image.convert("RGB")
                    output_image = np.array(image).astype(np.float32) / 255.0
                    output_image = torch.from_numpy(output_image).unsqueeze(0)
                return output_image, filename
            except (PIL.UnidentifiedImageError, OSError) as e:
                print(f"Warning: Skipping corrupted image file: {image_path} ({e})")
                return None, filename
        else:
            return None, None

    def run(self, folder_path, reset, reset_on_error, seed, loop_or_reset, include_extension, exclude_loaded_on_reset, output_alpha, start_index, use_manual_index, manual_index):
        random.seed(seed)

        if reset or not self.image_files or folder_path != self.prev_folder_path:
            self._load_image_files(folder_path)
            self.current_index = 0
            self.prev_folder_path = folder_path

        if not self.image_files:
            return (None, self.current_index, seed, None)

        while self.current_index < len(self.image_files):
            if use_manual_index == False :
                output_image, filename = self._load_image(folder_path, self.current_index + start_index, output_alpha)
            else :
                output_image, filename = self._load_image(folder_path, manual_index + start_index, output_alpha)
            if output_image is not None:
                break
            elif not reset_on_error:
                self.current_index += 1
            else:
                self._load_image_files(folder_path)
                if exclude_loaded_on_reset:
                    loaded_files = set(self.image_files[:self.current_index])
                    self.image_files = [f for f in self.image_files if f not in loaded_files]
                self.current_index = 0

        if self.current_index >= len(self.image_files):
            if loop_or_reset:
                self._load_image_files(folder_path)
                self.current_index = 0
            else:
                self.current_index = 0

            output_image, filename = self._load_image(folder_path, self.current_index, output_alpha)

        if not include_extension:
            filename = os.path.splitext(filename)[0]

        self.current_index += 1
        
        if use_manual_index == False :
            return_index = self.current_index + start_index - 1
        else :
            return_index = manual_index + start_index

        return (output_image, return_index, seed, filename)

NODE_CLASS_MAPPINGS = {
    "ImageSequenceLoader": ImageSequenceLoader
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageSequenceLoader": "Image Sequence Loader"
}
