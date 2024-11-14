import re

class TextCombiner:
    def __init__(self):
        self.text_log = []

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "separator": ("STRING", {"default": ","}),
                "remember_log": ("BOOLEAN", {"default": True}),
                "max_log": ("INT", {"default": 10, "min": 0, "max": 1000}),
                "allow_duplicate_log": ("BOOLEAN", {"default": True}),
                "use_regex": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "text_1": ("STRING", {"multiline": False, "default": "", "forceInput": True}),
                "text_2": ("STRING", {"multiline": False, "default": "", "forceInput": True}),
                "text_3": ("STRING", {"multiline": False, "default": "", "forceInput": True}),
                "text_4": ("STRING", {"multiline": False, "default": "", "forceInput": True}),
                "remove_text": ("STRING", {"multiline": False, "default": "", "forceInput": True}),
            },
        }

    RETURN_TYPES = ("STRING", "LIST", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("text", "text_log", "recent_text_1", "recent_text_2", "recent_text_3", "recent_text_4", "oldest_text")
    OUTPUT_NODE = True
    FUNCTION = "process_text"
    CATEGORY = "tksw_node"

    def process_text(self, text_1="", text_2="", text_3="", text_4="", separator=",", remember_log=True, max_log=10, allow_duplicate_log=False, remove_text="", use_regex=False):
        texts = [text_1, text_2, text_3, text_4]

        compiled_patterns = []
        if use_regex and remove_text:
            remove_patterns = [pattern.strip() for pattern in remove_text.split(",") if pattern.strip()]
            for pattern in remove_patterns:
                try:
                    compiled_patterns.append(re.compile(pattern))
                except re.error as e:
                    print(f"Invalid regex pattern '{pattern}': {e}")

        cleaned_texts = []
        for text in texts:
            split_text = text.split(separator)
            cleaned_parts = []
            for part in split_text:
                if use_regex and compiled_patterns:
                    cleaned_part = part
                    for compiled_pattern in compiled_patterns:
                        cleaned_part = compiled_pattern.sub("", cleaned_part)
                else:
                    remove_words = [word.strip() for word in remove_text.split(",") if word.strip()]
                    cleaned_part = part
                    for word in remove_words:
                        cleaned_part = cleaned_part.replace(word, "")
                cleaned_parts.append(cleaned_part)
            cleaned_text = separator.join(cleaned_parts)
            cleaned_texts.append(cleaned_text)

        combined_text = separator.join([text for text in cleaned_texts if text])

        pattern = r"(?<!\s)\s*[{separator}]{{2,}}\s*(?!\s)".format(separator=re.escape(separator))
        while re.search(pattern, combined_text):
            combined_text = re.sub(pattern, separator, combined_text)
        combined_text = combined_text.strip(separator)


        if remember_log:
            if allow_duplicate_log or combined_text not in self.text_log:
                self.text_log.append(combined_text)
                if len(self.text_log) > max_log:
                    self.text_log.pop(0)

            recent_texts = [""] * 4
            for i in range(min(4, len(self.text_log))):
                recent_texts[i] = self.text_log[-(i + 1)]

            oldest_text = ""
            if self.text_log:
                oldest_text = self.text_log[0]

            return (combined_text, list(self.text_log), *recent_texts, oldest_text)

        else:
            return (combined_text, [], *[""] * 4, "")
