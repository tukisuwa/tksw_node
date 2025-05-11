import os
import random
import torch
import collections
import comfy.utils
import comfy.sd 
from folder_paths import get_filename_list, supported_pt_extensions, get_full_path

LORA_SLOT_COUNT = 8
LORA_EXTENSIONS = [ext.lower() for ext in supported_pt_extensions]

class LoraSelector:
    def __init__(self):
        self.round_robin_index = 0
        self.last_candidate_list = None
        self.last_lora_folder = ""
        self.cached_folder_loras = []
        self.current_lora = None
        self.remaining_executions = 0
        self.loaded_lora_cache = collections.OrderedDict()
        self.lora_size_cache = {}
        self.current_cache_size_bytes = 0

    @classmethod
    def INPUT_TYPES(cls):
        slot_lora_list = [""] + get_filename_list("loras")
        inputs = {
            "required": {
                "strength_model": ("FLOAT", {"default": 1.00, "min": -10.00, "max": 10.00, "step": 0.01}),
                "strength_clip": ("FLOAT", {"default": 1.00, "min": -10.00, "max": 10.00, "step": 0.01}),
                "mode": (["random", "round-robin"], {"default": "random"}),
                "switch_interval": ("INT", {"default": 1, "min": 1, "max": 9999}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "reset_state": ("BOOLEAN", {"default": False}),
                "cache_limit_gb": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 128.0, "step": 0.1}), 
                "lora_folder": ("STRING", {"multiline": False, "default": ""}),
                **{f"lora_{i}": (slot_lora_list, {"default": ""}) for i in range(LORA_SLOT_COUNT)}
            },
            "optional": {
                 "model": ("MODEL",),
                 "clip": ("CLIP",),
            }
        }
        return inputs

    RETURN_TYPES = ("MODEL", "CLIP", "STRING")
    RETURN_NAMES = ("MODEL", "CLIP", "selected_lora_name")
    FUNCTION = "select_and_apply_lora"
    CATEGORY = "tksw_node"

    def _scan_lora_folder(self, folder_path):
        lora_files = []
        try:
            print(f"[LoraSelector] Scanning folder for LoRA files: {folder_path}")
            if not os.path.isdir(folder_path):
                 print(f"[LoraSelector] Warning: Specified LoRA folder does not exist or is not a directory: {folder_path}")
                 return []
            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)
                if os.path.isfile(file_path):
                    if any(filename.lower().endswith(ext) for ext in LORA_EXTENSIONS):
                        lora_files.append(filename)
            print(f"[LoraSelector] Found {len(lora_files)} LoRA files in folder.")
            return sorted(lora_files)
        except Exception as e:
            print(f"[LoraSelector] Error scanning folder '{folder_path}': {e}")
            return []

    def select_and_apply_lora(self, strength_model, strength_clip, mode, switch_interval, seed, reset_state, cache_limit_gb, lora_folder, model=None, clip=None, **kwargs):
        current_folder_loras = []
        clean_lora_folder = lora_folder.strip()
        if clean_lora_folder:
            if clean_lora_folder != self.last_lora_folder:
                self.cached_folder_loras = self._scan_lora_folder(clean_lora_folder)
                self.last_lora_folder = clean_lora_folder
            current_folder_loras = self.cached_folder_loras
        else:
            if self.last_lora_folder != "": print("[LoraSelector] LoRA folder cleared.")
            self.cached_folder_loras = []
            self.last_lora_folder = ""
            current_folder_loras = []
        slot_loras = []
        for i in range(LORA_SLOT_COUNT):
            lora_name = kwargs.get(f"lora_{i}", "")
            if lora_name and lora_name != "": slot_loras.append(lora_name)
        candidate_loras = sorted(list(set(slot_loras + current_folder_loras)))

        selected_lora_name = "None"
        chosen_lora = None
        num_candidates = len(candidate_loras)

        if candidate_loras:
            needs_reset = False
            reset_reason = ""
            if reset_state:
                needs_reset = True
                reset_reason = "Manual reset requested."
            elif self.last_candidate_list is not None and candidate_loras != self.last_candidate_list:
                needs_reset = True
                reset_reason = "LoRA candidate list changed."
            elif self.current_lora is not None and self.current_lora not in candidate_loras:
                needs_reset = True
                reset_reason = f"Current LoRA '{self.current_lora}' not in candidate list."

            if needs_reset:
                print(f"[LoraSelector] State reset triggered: {reset_reason}")
                self.current_lora = None
                self.remaining_executions = 0
                self.round_robin_index = 0
                if reset_state:
                     print("[LoraSelector] Clearing LoRA data cache due to manual reset.")
                     self.loaded_lora_cache.clear()
                     self.lora_size_cache.clear()
                     self.current_cache_size_bytes = 0
                self.last_candidate_list = list(candidate_loras) 
            else:
                 if self.last_candidate_list is None or candidate_loras != self.last_candidate_list:
                      self.last_candidate_list = list(candidate_loras)

            current_switch_interval = max(1, switch_interval)
            if self.remaining_executions > 0 and self.current_lora is not None:
                chosen_lora = self.current_lora
                self.remaining_executions -= 1
                print(f"[LoraSelector] Continuing LoRA: '{chosen_lora}' (Remaining: {self.remaining_executions} / Interval: {current_switch_interval})")
            else:
                print(f"[LoraSelector] Switching LoRA (Interval: {current_switch_interval})")
                if mode == "round-robin":
                    current_index = self.round_robin_index % num_candidates
                    chosen_lora = candidate_loras[current_index]
                    self.round_robin_index += 1
                    print(f"[LoraSelector] Mode: round-robin switch (Index: {current_index}, Next RR Base: {self.round_robin_index})")
                elif mode == "random":
                    random.seed(seed)
                    available_choices = [lora for lora in candidate_loras if lora != self.current_lora]
                    if not available_choices and num_candidates > 0: available_choices = candidate_loras
                    chosen_lora = random.choice(available_choices)
                    print(f"[LoraSelector] Mode: random switch (Seed: {seed})")
                else:
                    print(f"[LoraSelector] Warning: Unknown mode '{mode}'. Falling back to random.")
                    random.seed(seed)
                    chosen_lora = random.choice(candidate_loras)

                self.current_lora = chosen_lora
                self.remaining_executions = current_switch_interval - 1
                print(f"[LoraSelector] Switched to LoRA: '{chosen_lora}' (Use for {current_switch_interval} times, Remaining: {self.remaining_executions})")


            output_model = model
            output_clip = clip
            cache_enabled = cache_limit_gb > 0
            cache_limit_bytes = int(cache_limit_gb * (1024**3)) if cache_enabled else 0

            if chosen_lora:
                selected_lora_name = chosen_lora
                if output_model is None and output_clip is None:
                    print(f"[LoraSelector] Warning: LoRA '{chosen_lora}' selected, but no Model or Clip input provided.")
                    return (output_model, output_clip, selected_lora_name)

                print(f"[LoraSelector] Applying LoRA: '{chosen_lora}'")
                effective_strength_model = strength_model if output_model is not None else 0.0
                effective_strength_clip = strength_clip if output_clip is not None else 0.0
                print(f"[LoraSelector] Effective Strength - Model: {effective_strength_model if model else 'N/A'}, Clip: {effective_strength_clip if clip else 'N/A'}")

                try:
                    lora_data = None
                    cache_hit = False

                    if cache_enabled and chosen_lora in self.loaded_lora_cache:
                        lora_data = self.loaded_lora_cache[chosen_lora]
                        self.loaded_lora_cache.move_to_end(chosen_lora)
                        print(f"[LoraSelector] Using cached LoRA data for '{chosen_lora}'. Cache Size: {self.current_cache_size_bytes / (1024**3):.2f}/{cache_limit_gb:.1f} GB")
                        cache_hit = True

                    if not cache_hit:
                        lora_path = get_full_path("loras", chosen_lora)
                        if not lora_path:
                            raise FileNotFoundError(f"LoRA file not found in known paths: {chosen_lora}")

                        print(f"[LoraSelector] Loading LoRA data from: {lora_path}")
                        lora_data = comfy.utils.load_torch_file(lora_path, safe_load=True)

                        if cache_enabled and lora_data is not None:
                            try:
                                lora_file_size = os.path.getsize(lora_path)

                                while self.current_cache_size_bytes + lora_file_size > cache_limit_bytes and self.loaded_lora_cache:
                                    oldest_key, _ = self.loaded_lora_cache.popitem(last=False)
                                    removed_size = self.lora_size_cache.pop(oldest_key, 0) # サイズ情報も削除
                                    self.current_cache_size_bytes -= removed_size
                                    print(f"[LoraSelector] Cache evicted '{oldest_key}' (Size: {removed_size / (1024**2):.1f} MB) to make space. New Size: {self.current_cache_size_bytes / (1024**3):.2f}/{cache_limit_gb:.1f} GB")

                                if self.current_cache_size_bytes + lora_file_size <= cache_limit_bytes:
                                    self.loaded_lora_cache[chosen_lora] = lora_data 
                                    self.lora_size_cache[chosen_lora] = lora_file_size 
                                    self.current_cache_size_bytes += lora_file_size
                                    print(f"[LoraSelector] Cached LoRA data for '{chosen_lora}'. Cache Size: {self.current_cache_size_bytes / (1024**3):.2f}/{cache_limit_gb:.1f} GB. Items: {len(self.loaded_lora_cache)}")
                                else:
                                    print(f"[LoraSelector] Warning: LoRA '{chosen_lora}' (Size: {lora_file_size / (1024**2):.1f} MB) is too large for cache limit ({cache_limit_gb:.1f} GB). Not caching.")
                            except FileNotFoundError:
                                print(f"[LoraSelector] Warning: Could not get file size for '{lora_path}'. Caching without size check might be unreliable.")
                                if cache_enabled: 
                                     self.loaded_lora_cache[chosen_lora] = lora_data
                                     print(f"[LoraSelector] Cached LoRA data for '{chosen_lora}' (size unknown).")
                            except Exception as e:
                                print(f"[LoraSelector] Error during cache management for '{chosen_lora}': {e}")

                    if lora_data is not None:
                         applied_model, applied_clip = comfy.sd.load_lora_for_models(
                             output_model, output_clip, lora_data,
                             effective_strength_model, effective_strength_clip
                         )
                         output_model = applied_model
                         output_clip = applied_clip
                    else:
                         print(f"[LoraSelector] Warning: LoRA data for '{chosen_lora}' is None. Skipping application.")


                except Exception as e:
                    print(f"[LoraSelector] ERROR during LoRA loading or application for '{chosen_lora}': {e}")
                    selected_lora_name = f"ERROR applying {chosen_lora}"
                    return (model, clip, selected_lora_name)
            else:
                 print("[LoraSelector] No LoRA was chosen.")

        else: 
            print("[LoraSelector] No LoRA candidates found. Passing through inputs.")
            self.current_lora = None
            self.remaining_executions = 0
            self.round_robin_index = 0
            self.last_candidate_list = None

        return (output_model, output_clip, selected_lora_name)