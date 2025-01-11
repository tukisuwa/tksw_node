import re
import random
import os

class RandomWordReplacer:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "input_text": ("STRING", {"multiline": True, "default": "", "forceInput": True}),
                "replace_specs": ("STRING", {"multiline": True, "default": ""}),
                "replace_specs_file": ("STRING", {"multiline": False, "default": ""}),
                "replace_specs_folder": ("STRING", {"multiline": False, "default": ""}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("processed_text",)
    FUNCTION = "replace_words"
    CATEGORY = "tksw_node"

    def replace_words(self, seed, input_text=None, replace_specs_file="", replace_specs="", replace_specs_folder=""):
        if seed:
            random.seed(seed)

        if not input_text:
            return ("",)

        word_groups = []

        # 1. フォルダ処理 (replace_specs_folder)
        if replace_specs_folder:
            try:
                for filename in os.listdir(replace_specs_folder):
                    if filename.endswith((".txt", ".csv")):
                        with open(os.path.join(replace_specs_folder, filename), "r", encoding="utf-8") as f:
                            words = [line.strip() for line in f if line.strip()]
                            word_groups.append(words)
            except FileNotFoundError:
                return (f"Error: Folder not found: ", )

        # 2. ファイル処理 (replace_specs_file)
        if replace_specs_file:
            try:
                with open(replace_specs_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if line:
                            words = [word.strip() for word in line.split(',')]
                            if len(words) > 1:
                                word_groups.append(words)
            except FileNotFoundError:
                return (f"Error: File not found: ", )

        # 3. 複数行文字列処理 (replace_specs)
        if replace_specs:
            for line in replace_specs.splitlines():
                if line:
                    words = [word.strip() for word in line.split(',')]
                    if len(words) > 1:
                        word_groups.append(words)

        processed_text = ""
        for line in input_text.splitlines():
            processed_line = self.process_line(line, word_groups)
            processed_text += processed_line + "\n"
        return (processed_text.strip(),)

    def process_line(self, line, word_groups):
        processed_line = line
        for group in word_groups:
            for i, word in enumerate(group):
                if word:
                    while word in processed_line:
                        # グループ内の単語数が1つだけの場合は、同じ単語で置換
                        if len(group) == 1:
                            replacement = word
                        else:
                            replacement = random.choice([w for w in group if w != word])
                        processed_line = processed_line.replace(word, replacement, 1)
        return processed_line