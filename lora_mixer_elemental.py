import torch
import comfy.utils
import folder_paths
import comfy.sd
from safetensors.torch import safe_open
import random
import os
import json

class LoraMixerElemental:
    MAX_LORAS = 8

    @classmethod
    def INPUT_TYPES(cls):
        lora_names = ["None"] + folder_paths.get_filename_list("loras")
        input_types = {
            "required": {
                "model_strength": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
                "clip_strength": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "save_mixed_lora": (["Off", "First Only", "All"], {"default": "Off"}),
                "save_name": ("STRING", {"default": "mixed_lora"}),
                "key_selection": (["All Available Keys", "Common Keys Only"], {"default": "All Available Keys"}),
                "key_strength_randomization": (["Off", "Per Key"], {"default": "Off"}),
                "key_strength_min": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "key_strength_max": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "multi_mix": (["Off", "On", "Reuse Keys"], {"default": "Off"}),
                "num_mix_passes": ("INT", {"default": 2, "min": 2, "max": 10}),
                "mix_passes_strength_randomization": (["Off", "On"], {"default": "Off"}),
                "mix_pass_strength_min": ("FLOAT", {"default": 0.5, "min": -10.0, "max": 10.0, "step": 0.01}),
                "mix_pass_strength_max": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
            },
            "optional": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
            }
        }
        for i in range(1, cls.MAX_LORAS + 1):
            input_types["optional"][f"lora_name_{i}"] = (lora_names, {"default": "None"})
        return input_types

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    RETURN_TYPES = ("MODEL", "CLIP", "LORA", "STRING", "STRING")
    RETURN_NAMES = ("model", "clip", "mixed_lora", "lora_keys", "all_lora_keys")
    FUNCTION = "mix_loras"
    CATEGORY = "tksw_node"

    def mix_loras(self, model_strength, clip_strength, seed, save_mixed_lora, save_name, key_selection,
                  key_strength_randomization, key_strength_min, key_strength_max, multi_mix, num_mix_passes,
                  mix_passes_strength_randomization, mix_pass_strength_min, mix_pass_strength_max, model=None, clip=None, **kwargs):

        if model is None and clip is None:
            raise ValueError("Either 'model' or 'clip' must be provided.")

        lora_names = [
            kwargs[key] for key in kwargs
            if key.startswith("lora_name_") and kwargs[key] != "None"
        ]
        if not lora_names:
            return (model, clip, None, "", "{}") 

        loras = []
        lora_name_map = {}
        lora_dims = {}

        for i, lora_name in enumerate(lora_names):
            lora_path = folder_paths.get_full_path("loras", lora_name)
            try:
                lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
                loras.append(lora)
                lora_name_map[f"lora{i+1}"] = lora_name
                lora_dims[f"lora{i+1}"] = {}
                for key, value in lora.items():
                    if isinstance(value, torch.Tensor):
                        lora_dims[f"lora{i+1}"][key] = value.shape
            except Exception as e:
                print(f"Error loading LoRA {lora_name}: {e}")
                continue

        if not loras:
             return (model, clip, None, "", "{}")

        if key_selection == "Common Keys Only":
            common_keys = set(loras[0].keys())
            for lora in loras[1:]:
                common_keys.intersection_update(lora.keys())
            filtered_keys = {key for key in common_keys if key.endswith((".lora_down.weight", ".lora_up.weight"))}
        elif key_selection == "All Available Keys":
            all_keys = set()
            for lora in loras:
                all_keys.update(lora.keys())
            filtered_keys = {key for key in all_keys if key.endswith((".lora_down.weight", ".lora_up.weight"))}
        else:
            raise ValueError(f"Invalid key_selection_mode: {key_selection}")

        random.seed(seed)

        current_model = model
        current_clip = clip
        first_mixed_lora = None
        first_lora_keys_string = ""
        all_lora_keys = {}

        for mix_num in range(num_mix_passes if multi_mix != "Off" else 1):
            mixed_lora = {}
            key_source_map = {}

            temp_filtered_keys = filtered_keys if multi_mix == "Reuse Keys" else filtered_keys.copy()

            for down_key in list(temp_filtered_keys):
                if not down_key.endswith(".lora_down.weight"):
                    continue

                up_key = down_key.replace(".lora_down.weight", ".lora_up.weight")
                if up_key not in temp_filtered_keys:
                    continue

                available_down_loras = []
                for i, lora in enumerate(loras):
                    if down_key in lora:
                        available_down_loras.append((i, lora))
                if not available_down_loras:
                    continue

                selected_down_index, selected_down_lora = random.choice(available_down_loras)

                available_up_loras = []
                for i, lora in enumerate(loras):
                    if up_key in lora:
                        if lora_dims[f"lora{i+1}"][up_key] == lora_dims[f"lora{selected_down_index+1}"][up_key]:
                            available_up_loras.append((i, lora))

                if not available_up_loras:
                    temp_filtered_keys.discard(down_key)
                    temp_filtered_keys.discard(up_key)
                    continue

                selected_up_index, selected_up_lora = random.choice(available_up_loras)

                if key_strength_randomization == "Per Key":
                    strength = random.uniform(key_strength_min, key_strength_max)
                    mixed_lora[down_key] = selected_down_lora[down_key] * strength
                    mixed_lora[up_key] = selected_up_lora[up_key] * strength
                else:
                    mixed_lora[down_key] = selected_down_lora[down_key]
                    mixed_lora[up_key] = selected_up_lora[up_key]

                key_source_map[down_key] = lora_name_map[f"lora{selected_down_index + 1}"]
                key_source_map[up_key] = lora_name_map[f"lora{selected_up_index + 1}"]

                if multi_mix != "Reuse Keys":
                    temp_filtered_keys.discard(down_key)
                    temp_filtered_keys.discard(up_key)
            
            if not mixed_lora:
                continue

            if multi_mix != "Off" and mix_passes_strength_randomization == "On":
                mix_strength_model = random.uniform(mix_pass_strength_min, mix_pass_strength_max)
                mix_strength_clip = random.uniform(mix_pass_strength_min, mix_pass_strength_max)
            else:
                mix_strength_model = model_strength
                mix_strength_clip = clip_strength

            current_model, current_clip = comfy.sd.load_lora_for_models(current_model, current_clip, mixed_lora, mix_strength_model, mix_strength_clip)

            sorted_key_source_pairs = sorted(key_source_map.items(), key=lambda item: item[0])
            all_lora_keys[f"mix_pass_{mix_num + 1}"] = {key: source for key, source in sorted_key_source_pairs}


            if mix_num == 0:
                first_mixed_lora = mixed_lora
                first_lora_keys_string = "\n".join([f"{key}:{source}" for key, source in sorted_key_source_pairs])

            if save_mixed_lora == "All":
                current_save_name = f"{save_name}_{mix_num + 1}"
                if not current_save_name.endswith(".safetensors"):
                    current_save_name += ".safetensors"
                lora_path = os.path.join(folder_paths.get_folder_paths("loras")[0], current_save_name)
                try:
                    lora_to_save = {k: v for k, v in mixed_lora.items() if isinstance(v, torch.Tensor)}
                    comfy.utils.save_torch_file(lora_to_save, lora_path)
                    print(f"Mixed LoRA {mix_num + 1} saved to: {lora_path}")
                except Exception as e:
                    print(f"Error saving mixed LoRA {mix_num + 1}: {e}")

        if save_mixed_lora == "First Only" and first_mixed_lora:
            if not save_name.endswith(".safetensors"):
                save_name += ".safetensors"
            lora_path = os.path.join(folder_paths.get_folder_paths("loras")[0], save_name)
            try:
                lora_to_save = {k: v for k, v in first_mixed_lora.items() if isinstance(v, torch.Tensor)}
                comfy.utils.save_torch_file(lora_to_save,lora_path)
                print(f"First mixed LoRA saved to: {lora_path}")
            except Exception as e:
                print(f"Error saving first mixed LoRA: {e}")

        return (current_model, current_clip, first_mixed_lora, first_lora_keys_string, json.dumps(all_lora_keys, indent=4))