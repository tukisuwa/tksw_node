import os
import re
from PIL import Image
import numpy as np
import torch

class ImageLoaderSeedSync:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "folder_path": ("STRING", {"default": ""}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "reset": ("BOOLEAN", {"default": False}),  # ファイルリスト再読み込み用のリセットオプションを追加
                "sort_method": (["natural", "alphabetical"], {"default": "natural"}),
                "subfolder_depth": ("INT", {"default": 0, "min": 0, "max": 10}),  # サブフォルダ探索の深さ
                "output_path_mode": (["filename_only", "relative_path"], {"default": "filename_only"}),  # パス出力モード
                "output_alpha": ("BOOLEAN", {"default": False}),
                "include_extension": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "INT", "STRING")
    RETURN_NAMES = ("image", "index", "loop_count", "seed", "filename")
    FUNCTION = "run"
    CATEGORY = "tksw_node"

    def __init__(self):
        self.image_files = []
        self.prev_folder_path = ""
        self.prev_sort_method = ""
        self.prev_subfolder_depth = -1

    def _natural_sort_key(self, text):
        """自然順ソート用のキーを生成する"""
        def atoi(text):
            return int(text) if text.isdigit() else text.lower()
        return [atoi(c) for c in re.split(r'(\d+)', text)]

    def _collect_images_recursive(self, folder_path, current_depth, max_depth, base_path):
        """再帰的に画像ファイルを収集する"""
        image_files = []

        if not os.path.isdir(folder_path):
            return image_files

        try:
            entries = os.listdir(folder_path)
        except PermissionError:
            print(f"Warning: Permission denied for folder: {folder_path}")
            return image_files

        # 現在のフォルダ内の画像ファイルを収集
        for entry in entries:
            full_path = os.path.join(folder_path, entry)

            if os.path.isfile(full_path) and entry.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
                # ベースパスからの相対パスを保存
                relative_path = os.path.relpath(full_path, base_path)
                image_files.append(relative_path)

        # 指定された深さまでサブフォルダを探索
        if current_depth < max_depth:
            for entry in entries:
                full_path = os.path.join(folder_path, entry)
                if os.path.isdir(full_path):
                    image_files.extend(
                        self._collect_images_recursive(full_path, current_depth + 1, max_depth, base_path)
                    )

        return image_files

    def _load_image_files(self, folder_path, sort_method, subfolder_depth):
        """フォルダから画像ファイルのリストを読み込み、ソートしてキャッシュする"""
        if not os.path.isdir(folder_path):
            self.image_files = []
            return

        print(f"Reloading image list from: {folder_path} (sort: {sort_method}, depth: {subfolder_depth})") # 再読み込みが実行されたことを確認するためログ出力

        # サブフォルダ探索を使用して画像ファイルを収集
        image_files = self._collect_images_recursive(folder_path, 0, subfolder_depth, folder_path)

        if sort_method == "natural":
            self.image_files = sorted(image_files, key=self._natural_sort_key)
        else:  # alphabetical
            self.image_files = sorted(image_files)

        if not self.image_files:
            print(f"Warning: No images found in folder: {folder_path}")

    def _load_image(self, folder_path, index, alpha, output_path_mode, include_extension):
        """指定されたインデックスの画像を読み込む"""
        if not self.image_files or not (0 <= index < len(self.image_files)):
            return None, None

        # self.image_files[index]は相対パス形式で保存されている
        relative_path = self.image_files[index]
        image_path = os.path.join(folder_path, relative_path)

        # 出力するファイル名を決定
        if output_path_mode == "relative_path":
            # 相対パス形式（サブフォルダ名/ファイル名）
            output_filename = relative_path.replace(os.sep, '/')  # パス区切りを統一
        else:
            # ファイル名のみ
            output_filename = os.path.basename(relative_path)

        # 拡張子の処理
        if not include_extension:
            output_filename = os.path.splitext(output_filename)[0]

        try:
            with Image.open(image_path) as image:
                if not alpha:
                    image = image.convert("RGB")
                output_image = np.array(image).astype(np.float32) / 255.0
                output_image = torch.from_numpy(output_image).unsqueeze(0)
            return output_image, output_filename
        except Exception as e:
            print(f"Warning: Failed to load image {image_path}. Error: {e}")
            return None, output_filename

    def run(self, folder_path, seed, reset, sort_method, subfolder_depth, output_path_mode, output_alpha, include_extension):
        # [変更点] resetがTrueの場合、またはフォルダパス、ソート方法、サブフォルダ深さが変更された場合にファイルリストを再読み込み
        if reset or folder_path != self.prev_folder_path or sort_method != self.prev_sort_method or subfolder_depth != self.prev_subfolder_depth:
            self._load_image_files(folder_path, sort_method, subfolder_depth)
            self.prev_folder_path = folder_path
            self.prev_sort_method = sort_method
            self.prev_subfolder_depth = subfolder_depth

        if not self.image_files:
            return (None, -1, -1, seed, "No images found")

        num_images = len(self.image_files)
        index = seed % num_images
        loop_count = seed // num_images

        output_image, filename = self._load_image(folder_path, index, output_alpha, output_path_mode, include_extension)

        if output_image is None:
            return (None, index, loop_count, seed, f"Failed to load: {filename or 'unknown'}")

        return (output_image, index, loop_count, seed, filename)