import json


class TextFastPreview:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"forceInput": True, "multiline": True, "default": ""}),
            },
            "optional": {
                "save_history": ("BOOLEAN", {"default": True}),
                "history_json": ("STRING", {"multiline": True, "default": ""}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("text", "history_text", "history_json")
    FUNCTION = "preview"
    CATEGORY = "tksw_node"
    OUTPUT_NODE = True
    DESCRIPTION = "Fast text preview with history switching in the UI."

    def preview(self, text, save_history=True, history_json=""):
        if text is None:
            text = ""
        history = []
        try:
            data = json.loads(history_json or "{}")
            if isinstance(data.get("history"), list):
                history = [str(item) for item in data["history"]]
        except Exception:
            history = []

        if text and (not history or history[-1] != text):
            history.append(str(text))
            history = history[-20:]

        history_json_out = json.dumps(
            {"history": history, "index": max(0, len(history) - 1) if history else -1},
            ensure_ascii=False,
        ) if save_history else ""

        return {
            "ui": {"preview_text": [text]},
            "result": (text, "\n\n---\n\n".join(history), history_json_out),
        }


NODE_CLASS_MAPPINGS = {
    "TextFastPreview": TextFastPreview,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TextFastPreview": "Text Fast Preview",
}
