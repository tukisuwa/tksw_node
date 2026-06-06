import torch
import comfy.utils
import folder_paths
import comfy.sd
from safetensors.torch import safe_open, save_file
import json
import io
import math
import os
import re


class LoraLoaderElemental:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
                "strength_clip": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
            },
            "optional": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "lora_strength_string": ("STRING", {"multiline": True, "default": ""}),
                "save_lora": ("BOOLEAN", {"default": False}),
                "save_name": ("STRING", {"default": "processed_lora"}),
                "remove_unspecified_keys": ("BOOLEAN", {"default": False}),
                "remove_zero_strength_keys": ("BOOLEAN", {"default": False}),  
                "match_mode": (["prefix", "contains", "regex"], {"default": "prefix"}),
            }
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    RETURN_TYPES = ("MODEL", "CLIP", "LORA", "STRING", "STRING")
    OPTIONAL_INPUTS = ("MODEL", "CLIP")
    RETURN_NAMES = ("model", "clip", "processed_lora", "metadata", "lora_keys")

    FUNCTION = "load_lora"
    CATEGORY = "tksw_node"

    def _parse_strength_string(self, strength_string):
        lora_strengths = {}  
        with io.StringIO(strength_string) as f:
            for index, line in enumerate(f):
                line = line.strip()
                if line and "=" in line:
                    try:
                        key, value = line.split("=", 1)
                        key = key.strip()
                        value = float(value.strip())
                        lora_strengths[key] = (value, index)
                    except ValueError:
                        print(f"Invalid line in strength string: ")
        return lora_strengths

    def _save_processed_lora(self, lora, save_name):
        if not save_name.endswith(".safetensors"):
            save_name += ".safetensors"
        lora_path = os.path.join(folder_paths.get_folder_paths("loras")[0], save_name)

        metadata = lora.pop("metadata", {}) if isinstance(lora, dict) else {}
        try:
            save_file(lora, lora_path, metadata)
            print(f"Processed LoRA saved to: {lora_path}") 
        except Exception as e:
            print(f"Error saving processed LoRA: {e}")  
        if metadata:
            lora["metadata"] = metadata

    def _get_lora_keys_string(self, lora):
        if not isinstance(lora, dict):
            return ""

        keys = [key for key in lora if key != "metadata"]
        candidates = set()
        for key in keys:
            parts = key.split(".")
            for depth in (1, 2, 3, 4):
                if len(parts) >= depth:
                    candidates.add(".".join(parts[:depth]))

            match = re.search(r"(diffusion_model\.blocks\.\d+)", key)
            if match:
                candidates.add(match.group(1))
            match = re.search(r"(diffusion_model\.llm_adapter\.blocks\.\d+)", key)
            if match:
                candidates.add(match.group(1))
            match = re.search(r"(transformer(?:\.single_transformer_blocks|\.transformer_blocks)\.\d+)", key)
            if match:
                candidates.add(match.group(1))
            match = re.search(r"((?:lora_)?(?:unet_)?(?:input|output)_blocks?[_\.]\d+)", key)
            if match:
                candidates.add(match.group(1))
            match = re.search(r"((?:single|double)_(?:transformer_)?blocks?[_\.]\d+)", key)
            if match:
                candidates.add(match.group(1))

        return "\n".join(sorted(candidates))

    def _matches_key(self, pattern, lora_key, match_mode="prefix"):
        if match_mode == "regex":
            try:
                return re.search(pattern, lora_key) is not None
            except re.error as e:
                print(f"Invalid regular expression '{pattern}': {e}")
                return False
        if match_mode == "contains":
            return pattern in lora_key
        return lora_key.startswith(pattern)

    def _elemental_tensor_strength(self, strength, key):
        sign = -1.0 if strength < 0 else 1.0
        scale = math.sqrt(abs(strength))
        if key.endswith(".lora_up.weight"):
            return scale * sign
        return scale

    def load_lora(self, lora_name, strength_model, strength_clip, model=None, clip=None,
                 lora_strength_string="", save_lora=False, save_name="processed_lora",
                 remove_unspecified_keys=False, remove_zero_strength_keys=False, match_mode="prefix", regex_mode=False): 

        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)

        if model is None and clip is None:
            raise ValueError("Either 'model' or 'clip' must be provided.")

        if strength_model == 0 and strength_clip == 0:
            try:
                with safe_open(lora_path, framework="pt", device="cpu") as f:
                    lora_metadata = f.metadata()
                lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
            except Exception as e:
                print(f"Error reading LoRA metadata or loading: {e}") 
                lora_metadata = {}
                lora = {}
            lora_keys_string = self._get_lora_keys_string(lora)
            return (model, clip, None, json.dumps(lora_metadata, indent=4), lora_keys_string)

        try:
            with safe_open(lora_path, framework="pt", device="cpu") as f:
                lora_metadata = f.metadata()
            lora = comfy.utils.load_torch_file(lora_path, safe_load=True)

        except Exception as e:
            print(f"Error loading LoRA file: {e}") 
            return (model, clip, None, None, "")

        lora_strengths = {}
        if lora_strength_string:
            lora_strengths = self._parse_strength_string(lora_strength_string)

        extended_lora = {}
        for key, value in lora.items():
            if key == "metadata":
                extended_lora[key] = {"value": value}
            elif key.endswith((".lora_down.weight", ".lora_up.weight")):
                extended_lora[key] = {"strength": None, "specified": False}
            else:
                extended_lora[key] = {"strength": None, "specified": False}

        effective_match_mode = "regex" if regex_mode else match_mode
        for strength_key, (strength, index) in lora_strengths.items():
            for lora_key in list(extended_lora.keys()):
                if self._matches_key(strength_key, lora_key, match_mode=effective_match_mode):
                    extended_lora[lora_key]["strength"] = strength
                    extended_lora[lora_key]["specified"] = True

        new_lora = {}
        for key, data in extended_lora.items():
            if key == "metadata":
                new_lora[key] = data["value"]
                continue

            if key.endswith((".lora_down.weight", ".lora_up.weight")):
                if data["specified"]:
                    if remove_zero_strength_keys and data["strength"] == 0: 
                        continue
                    new_lora[key] = lora[key] * self._elemental_tensor_strength(data["strength"], key)

                elif not remove_unspecified_keys:
                    new_lora[key] = lora[key]
            else:  
                if not remove_unspecified_keys:
                    new_lora[key] = lora[key]


        model_lora, clip_lora = comfy.sd.load_lora_for_models(model, clip, new_lora, strength_model, strength_clip)
        lora_metadata_string = json.dumps(lora_metadata, indent=4)

        if save_lora:
            self._save_processed_lora(new_lora, save_name)

        lora_keys_string = self._get_lora_keys_string(new_lora)

        return (model_lora, clip_lora, new_lora, lora_metadata_string, lora_keys_string)


class LoraLoaderElementalUI(LoraLoaderElemental):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
                "strength_clip": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
                "elemental_settings_json": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Internal dynamic UI state for pattern-based LoRA strength rules.",
                }),
            },
            "optional": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "save_lora": ("BOOLEAN", {"default": False}),
                "save_name": ("STRING", {"default": "processed_lora"}),
                "remove_unspecified_keys": ("BOOLEAN", {"default": False}),
                "remove_zero_strength_keys": ("BOOLEAN", {"default": False}),
                "match_mode": (["prefix", "contains", "regex"], {"default": "prefix"}),
            }
        }

    CATEGORY = "tksw_node"
    DESCRIPTION = "LoRA Loader Elemental with a dynamic rule editor UI."

    def _settings_json_to_strength_string(self, settings_json):
        try:
            data = json.loads(settings_json or "{}")
        except Exception:
            data = {}
        items = data.get("items") or []
        lines = []
        for item in items:
            if not isinstance(item, dict):
                continue
            pattern = str(item.get("pattern", "")).strip()
            if not pattern:
                continue
            try:
                strength = float(item.get("strength", 1.0))
            except Exception:
                strength = 1.0
            if not bool(item.get("enabled", True)):
                continue
            lines.append(f"{pattern} = {strength:g}")
        return "\n".join(lines)

    def load_lora(self, lora_name, strength_model, strength_clip, elemental_settings_json="", model=None, clip=None,
                 save_lora=False, save_name="processed_lora",
                 remove_unspecified_keys=False, remove_zero_strength_keys=False, match_mode="prefix", regex_mode=False):
        lora_strength_string = self._settings_json_to_strength_string(elemental_settings_json)
        return super().load_lora(
            lora_name,
            strength_model,
            strength_clip,
            model=model,
            clip=clip,
            lora_strength_string=lora_strength_string,
            save_lora=save_lora,
            save_name=save_name,
            remove_unspecified_keys=remove_unspecified_keys,
            remove_zero_strength_keys=remove_zero_strength_keys,
            match_mode=match_mode,
            regex_mode=regex_mode,
        )
