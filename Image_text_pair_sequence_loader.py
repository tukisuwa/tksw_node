import os
import PIL
from PIL import Image
import numpy as np
import torch
import random

class ImageTextPairSequenceLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image_folder_path": ("STRING", {"default": ""}),
                "text_folder_path": ("STRING", {"default": ""}),
                "reset": ("BOOLEAN", {"default": False}),
                "loop_or_reset": ("BOOLEAN", {"default": False}),
                "reset_on_error": ("BOOLEAN", {"default": False}),
                "exclude_loaded_on_reset": ("BOOLEAN", {"default": False}),
                "output_alpha": ("BOOLEAN", {"default": False}),
                "include_extension": ("BOOLEAN", {"default": False}),
                "start_index": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "INT", "STRING")
    RETURN_NAMES = ("image", "text", "index", "filename")
    FUNCTION = "run"
    CATEGORY = "tksw_node"

    def __init__(self):
        self.current_index = 0
        self.image_files = []
        self.text_files = []
        self.common_basenames = [] 
        self.image_file_map = {}
        self.text_file_map = {}
        self.prev_image_folder_path = ""
        self.prev_text_folder_path = ""
        self.prev_start_index = 0

    def _load_files(self, image_folder_path, text_folder_path):
        try:
            self.image_files = sorted([
                f for f in os.listdir(image_folder_path)
                if os.path.isfile(os.path.join(image_folder_path, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp'))
            ])
        except FileNotFoundError:
            print(f"Warning: Image folder not found: {image_folder_path}")
            self.image_files = []
        except Exception as e:
            print(f"Error reading image folder {image_folder_path}: {e}")
            self.image_files = []

        try:
            self.text_files = sorted([
                f for f in os.listdir(text_folder_path)
                if os.path.isfile(os.path.join(text_folder_path, f)) and f.lower().endswith('.txt')
            ])
        except FileNotFoundError:
            print(f"Warning: Text folder not found: {text_folder_path}")
            self.text_files = []
        except Exception as e:
            print(f"Error reading text folder {text_folder_path}: {e}")
            self.text_files = []

        self.image_file_map = {os.path.splitext(f)[0]: f for f in self.image_files}
        self.text_file_map = {os.path.splitext(f)[0]: f for f in self.text_files}

        image_basenames = set(self.image_file_map.keys())
        text_basenames = set(self.text_file_map.keys())
        self.common_basenames = sorted(list(image_basenames & text_basenames))

        if not self.common_basenames:
             print("Warning: No common filenames (excluding extension) found between image and text folders.")


    def _load_image(self, folder_path, filename, alpha):
        image_path = os.path.join(folder_path, filename)
        try:
            with Image.open(image_path) as img:
                if alpha and img.mode != 'RGBA':
                    img = img.convert("RGBA")
                elif not alpha and img.mode != 'RGB':
                    img = img.convert("RGB")

                output_image = np.array(img).astype(np.float32) / 255.0
                output_image = torch.from_numpy(output_image).unsqueeze(0)
                return output_image
        except FileNotFoundError:
            print(f"Warning: Image file not found: {image_path}")
            return None
        except (PIL.UnidentifiedImageError, OSError) as e:
            print(f"Warning: Skipping corrupted or unsupported image file: {filename} ({e})")
            return None
        except Exception as e:
            print(f"Error loading image {filename}: {e}")
            return None

    def _load_text(self, folder_path, filename):
        text_path = os.path.join(folder_path, filename)
        try:
            with open(text_path, 'r', encoding='utf-8') as f:
                text_content = f.read()
            return text_content
        except FileNotFoundError:
            print(f"Warning: Text file not found: {text_path}")
            return None
        except Exception as e:
            print(f"Error loading text file {filename}: {e}")
            return None

    def run(self, image_folder_path, text_folder_path, reset, reset_on_error, seed, loop_or_reset, include_extension, exclude_loaded_on_reset, output_alpha, start_index):
        random.seed(seed) 

        if not image_folder_path or not text_folder_path:
             print("Error: Image folder path and Text folder path must be specified.")
             return (None, None, 0, None)

        needs_reload = (
            reset or
            not self.common_basenames or
            image_folder_path != self.prev_image_folder_path or
            text_folder_path != self.prev_text_folder_path or
            start_index != self.prev_start_index
        )

        if needs_reload:
            self._load_files(image_folder_path, text_folder_path)
            self.current_index = 0
            self.prev_image_folder_path = image_folder_path
            self.prev_text_folder_path = text_folder_path
            self.prev_start_index = start_index

            if self.common_basenames and start_index >= len(self.common_basenames):
                print(f"Warning: start_index ({start_index}) is out of bounds. Setting to last index ({len(self.common_basenames) - 1}).")
                start_index = len(self.common_basenames) - 1
            elif not self.common_basenames:
                start_index = 0

        if not self.common_basenames:
            print("Error: No image/text pairs found.")
            return (None, None, self.current_index, None)

        effective_index = self.current_index + start_index

        if effective_index >= len(self.common_basenames):
            if loop_or_reset:
                print("Looping back to start.")
                self._load_files(image_folder_path, text_folder_path) 
                self.current_index = 0
                start_index = 0 
                self.prev_start_index = 0 
                effective_index = 0
                if not self.common_basenames: 
                     print("Error: No image/text pairs found after reload.")
                     return (None, None, self.current_index, None)
            else:
                print("Reached end of sequence.")
                return (None, None, self.current_index, None)

        output_image = None
        output_text = None
        current_basename = None
        loaded_successfully = False

        original_start_index = self.current_index 

        while effective_index < len(self.common_basenames):
            current_basename = self.common_basenames[effective_index]
            image_filename = self.image_file_map.get(current_basename)
            text_filename = self.text_file_map.get(current_basename)

            if not image_filename or not text_filename:
                 print(f"Error: Internal inconsistency. Basename '{current_basename}' not found in file maps.")
            else:
                output_image = self._load_image(image_folder_path, image_filename, output_alpha)
                output_text = self._load_text(text_folder_path, text_filename)

                if output_image is not None and output_text is not None:
                    loaded_successfully = True
                    break 
                else:
                    print(f"Failed to load pair for basename: {current_basename}")
                    if reset_on_error:
                        print("Resetting sequence due to error.")
                        if exclude_loaded_on_reset:
                             print("Excluding previously loaded files on reset.")
                             loaded_basenames = set(self.common_basenames[:original_start_index + start_index])
                             self._load_files(image_folder_path, text_folder_path)
                             self.common_basenames = [b for b in self.common_basenames if b not in loaded_basenames]
                        else:
                             self._load_files(image_folder_path, text_folder_path)

                        self.current_index = 0
                        start_index = 0 
                        self.prev_start_index = 0
                        effective_index = 0

                        if not self.common_basenames: 
                             print("Error: No image/text pairs remaining after reset.")
                             return (None, None, 0, None)
                        continue 

                    else:
                        print("Skipping current pair due to error.")
                        self.current_index += 1
                        effective_index = self.current_index + start_index
                        if effective_index >= len(self.common_basenames):
                             print("Reached end of sequence after skipping error.")
                             return (None, None, self.current_index, None) 

        if not loaded_successfully:
             print("Error: Could not successfully load any remaining image/text pair.")
             return (None, None, self.current_index, None)


        output_filename = current_basename
        if include_extension:
            image_filename = self.image_file_map.get(current_basename)
            if image_filename:
                 output_filename = image_filename

        final_index = effective_index
        self.current_index += 1

        return (output_image, output_text, final_index, output_filename)