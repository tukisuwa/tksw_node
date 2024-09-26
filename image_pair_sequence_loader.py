import os
from PIL import Image
import numpy as np
import torch
import random

class ImagePairSequenceLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "folder_path_A": ("STRING", {"default": ""}),
                "folder_path_B": ("STRING", {"default": ""}),
                "reset": ("BOOLEAN", {"default": False}),
                "loop_or_reset": ("BOOLEAN", {"default": False}),
                "reset_on_error": ("BOOLEAN", {"default": False}),
                "exclude_loaded_on_reset": ("BOOLEAN", {"default": False}),
                "output_alpha": ("BOOLEAN", {"default": False}),
                "include_extension": ("BOOLEAN", {"default": False}),
                "start_index": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "match_extension": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "INT", "STRING")
    RETURN_NAMES = ("image_A", "image_B", "index", "filename")
    FUNCTION = "run"
    CATEGORY = "image"

    def __init__(self):
        self.current_index = 0
        self.image_files_A = []
        self.image_files_B = []
        self.common_files = []
        self.prev_folder_path_A = ""
        self.prev_folder_path_B = ""
        self.prev_start_index = 0

    def _load_image_files(self, folder_path_A, folder_path_B, match_extension):
        self.image_files_A = sorted([
            f for f in os.listdir(folder_path_A)
            if os.path.isfile(os.path.join(folder_path_A, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
        ])
        self.image_files_B = sorted([
            f for f in os.listdir(folder_path_B)
            if os.path.isfile(os.path.join(folder_path_B, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
        ])

        if match_extension:
            self.common_files = sorted(list(set(self.image_files_A) & set(self.image_files_B)))
        else:
            filenames_A = [os.path.splitext(f)[0] for f in self.image_files_A]
            filenames_B = [os.path.splitext(f)[0] for f in self.image_files_B]
            common_filenames = sorted(list(set(filenames_A) & set(filenames_B)))
            self.common_files = [f + os.path.splitext(self.image_files_A[filenames_A.index(f)])[1] for f in common_filenames]

    def _load_image(self, folder_path, filename, alpha):
        image_path = os.path.join(folder_path, filename)
        try:
            with Image.open(image_path) as image:
                if alpha == False:
                    image = image.convert("RGB")
                output_image = np.array(image).astype(np.float32) / 255.0
                output_image = torch.from_numpy(output_image).unsqueeze(0)
            return output_image
        except (PIL.UnidentifiedImageError, OSError) as e:
            print(f"Warning: Skipping corrupted image file: {image_path} ({e})")
            return None

    def run(self, folder_path_A, folder_path_B, reset, reset_on_error, seed, loop_or_reset, include_extension, exclude_loaded_on_reset, output_alpha, start_index, match_extension):
        random.seed(seed)

        if not folder_path_B:
            folder_path_B = folder_path_A

        if reset or not self.common_files or folder_path_A != self.prev_folder_path_A or folder_path_B != self.prev_folder_path_B or start_index != self.prev_start_index:
            if folder_path_A == folder_path_B:
                self._load_image_files(folder_path_A, folder_path_A, match_extension)
                self.common_files = self.image_files_A
            else:
                self._load_image_files(folder_path_A, folder_path_B, match_extension)

            self.current_index = 0
            self.prev_folder_path_A = folder_path_A
            self.prev_folder_path_B = folder_path_B
            self.prev_start_index = start_index

            if start_index >= len(self.common_files):
                start_index = len(self.common_files) - 1

        if not self.common_files:
            return (None, None, self.current_index, None)

        if self.current_index + start_index >= len(self.common_files):
            if loop_or_reset:
                if folder_path_A == folder_path_B:
                    self._load_image_files(folder_path_A, folder_path_A, match_extension)
                    self.common_files = self.image_files_A
                else:
                    self._load_image_files(folder_path_A, folder_path_B, match_extension)
                self.current_index = 0
            else:
                self.current_index = 0

            filename = self.common_files[self.current_index + start_index]
            output_image_A = self._load_image(folder_path_A, filename, output_alpha)

            if folder_path_A == folder_path_B:
                output_image_B = output_image_A
            else:
                filename_B = os.path.splitext(filename)[0] + os.path.splitext(self.image_files_B[self.image_files_A.index(filename)])[1] if not match_extension else filename # 拡張子が異なる場合のファイル名
                output_image_B = self._load_image(folder_path_B, filename_B, output_alpha)

            if not include_extension:
                filename = os.path.splitext(filename)[0]

            self.current_index += 1

            return (output_image_A, output_image_B, self.current_index + start_index - 1, filename)

        while self.current_index + start_index < len(self.common_files):
            filename = self.common_files[self.current_index + start_index]
            output_image_A = self._load_image(folder_path_A, filename, output_alpha)

            if folder_path_A == folder_path_B:
                output_image_B = output_image_A
            else:
                filename_B = os.path.splitext(filename)[0] + os.path.splitext(self.image_files_B[self.image_files_A.index(filename)])[1] if not match_extension else filename
                output_image_B = self._load_image(folder_path_B, filename_B, output_alpha)


            if output_image_A is not None and output_image_B is not None:
                break
            elif not reset_on_error:
                self.current_index += 1
            else:
                if folder_path_A == folder_path_B:
                    self._load_image_files(folder_path_A, folder_path_A, match_extension)
                    self.common_files = self.image_files_A
                else:
                    self._load_image_files(folder_path_A, folder_path_B, match_extension)

                if exclude_loaded_on_reset:
                    loaded_files = set(self.common_files[:self.current_index])
                    self.common_files = [f for f in self.common_files if f not in loaded_files]
                self.current_index = 0

        if not include_extension:
            filename = os.path.splitext(filename)[0]

        self.current_index += 1

        return (output_image_A, output_image_B, self.current_index + start_index - 1, filename)