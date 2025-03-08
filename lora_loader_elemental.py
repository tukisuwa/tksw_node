import torch
import math
import comfy.utils
import folder_paths
import comfy.sd
from safetensors.torch import safe_open, save_file
import json
import io
import os


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
            for line in f:
                line = line.strip()
                if line and "=" in line:
                    try:
                        key, value = line.split("=", 1)
                        key = key.strip()
                        value = float(value.strip())
                        lora_strengths[key] = value
                    except ValueError:
                        print(f"Invalid line in strength string: {line}")
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

        prefixes = sorted(list(set(key.split(".")[0] for key in lora if key != "metadata")))
        return "\n".join(prefixes)

    def load_lora(self, lora_name, strength_model, strength_clip, model=None, clip=None,
                 lora_strength_string="", save_lora=False, save_name="processed_lora",
                 remove_unspecified_keys=False):

        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)

        if model is None and clip is None:
            raise ValueError("Either 'model' or 'clip' must be provided.")

        if strength_model == 0 and strength_clip == 0:
            try:
                with safe_open(lora_path, "pt", device="cpu") as f:
                    lora_metadata = f.metadata()
                    lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
            except Exception as e:
                print(f"Error reading LoRA metadata or loading: {e}")
                lora_metadata = {}
                lora = {}
            lora_keys_string = self._get_lora_keys_string(lora)
            return (model, clip, None, json.dumps(lora_metadata, indent=4), lora_keys_string)

        try:
            with safe_open(lora_path, "pt", device="cpu") as f:
                lora_metadata = f.metadata()
            lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
        except Exception as e:
            print(f"Error loading LoRA file: {lora_path} - {e}")
            return (model, clip, None, None, "")

        lora_strengths = {}
        if lora_strength_string:
            lora_strengths = self._parse_strength_string(lora_strength_string)

        if lora_strengths:
            target_prefixes = set()
            for lora_key in lora:
                for key_prefix in lora_strengths:
                    if key_prefix in lora_key:
                        prefix = lora_key.split(".")[0]
                        target_prefixes.add(prefix)
                        break

            for key in list(lora.keys()):
                for strength_key, strength in lora_strengths.items(): 
                    if key.startswith(strength_key) and key.endswith((".lora_down.weight", ".lora_up.weight")):
                        lora[key] *= math.sqrt(strength)
                        break 


            if remove_unspecified_keys:
                new_lora = {}
                for key, value in lora.items():
                    if key == "metadata" :
                        new_lora[key] = value
                        continue
                    for prefix in target_prefixes:
                        if key.startswith(prefix):
                            new_lora[key] = value
                            break

                lora = new_lora

        model_lora, clip_lora = comfy.sd.load_lora_for_models(model, clip, lora, strength_model, strength_clip)
        lora_metadata_string = json.dumps(lora_metadata, indent=4)

        if save_lora:
            self._save_processed_lora(lora, save_name)

        lora_keys_string = self._get_lora_keys_string(lora)

        return (model_lora, clip_lora, lora, lora_metadata_string, lora_keys_string)