import torch
import comfy.utils
import folder_paths
import comfy.sd
from safetensors.torch import safe_open, save_file
import json
import os
import numpy as np

class QuantizedLoraLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "quantization_bits": ("INT", {"default": 8, "min": 2, "max": 32, "step": 1}),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "strength_clip": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "quantization_iterations": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
                "stepwise_quantization": ("BOOLEAN", {"default": False}),
                "quantization_step_size": ("INT", {"default": 2, "min": 2, "max": 16, "step": 1}),
                "blend_mode": ("BOOLEAN", {"default": False}), 
                "blend_factor": ("FLOAT", {"default": 0.5, "min": -10.0, "max": 10.0, "step": 0.01}), 
            },
            "optional": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "save_quantized_lora": ("BOOLEAN", {"default": False}),
                "save_name": ("STRING", {"default": "quantized_lora"}),
            }
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    RETURN_TYPES = ("MODEL", "CLIP", "LORA", "STRING")
    RETURN_NAMES = ("model", "clip", "quantized_lora", "metadata")
    OPTIONAL_INPUTS = ("MODEL", "CLIP")

    FUNCTION = "load_and_quantize_lora"
    CATEGORY = "tksw_node"

    def _quantize_tensor(self, tensor, bits):
        original_dtype = tensor.dtype
        min_val = tensor.min()
        max_val = tensor.max()

        if max_val - min_val == 0:
            return torch.zeros_like(tensor, dtype=original_dtype)

        scale = (max_val - min_val) / (2**bits - 1)
        zero_point = torch.round(-min_val / scale)

        quantized_tensor = torch.clamp(torch.round(tensor / scale + zero_point), 0, 2**bits - 1)
        dequantized_tensor = (quantized_tensor - zero_point) * scale

        return dequantized_tensor.to(original_dtype)

    def _save_processed_lora(self, lora, save_name, quantization_bits, quantization_iterations, stepwise_quantization, quantization_step_size, blend_mode, blend_factor):
        if not save_name.endswith(".safetensors"):
            save_name += ".safetensors"
        base, ext = os.path.splitext(save_name)
        save_name = f"{base}_q{quantization_bits}_i{quantization_iterations}_s{stepwise_quantization}_ss{quantization_step_size}_b{blend_mode}_bf{blend_factor}{ext}"
        lora_path = os.path.join(folder_paths.get_folder_paths("loras")[0], save_name)

        metadata = lora.pop("metadata", {}) if isinstance(lora, dict) else {}

        if isinstance(metadata, dict):
            metadata["quantization_bits"] = str(quantization_bits)
            metadata["quantization_iterations"] = str(quantization_iterations)
            metadata["stepwise_quantization"] = str(stepwise_quantization)
            metadata["quantization_step_size"] = str(quantization_step_size)
            metadata["blend_mode"] = str(blend_mode) 
            metadata["blend_factor"] = str(blend_factor) 

            for k, v in metadata.items():
                if not isinstance(v, (str, int, float, list)):
                    metadata[k] = str(v)
        else:
            metadata = {
                "quantization_bits": str(quantization_bits),
                "quantization_iterations": str(quantization_iterations),
                "stepwise_quantization": str(stepwise_quantization),
                "quantization_step_size": str(quantization_step_size),
                "blend_mode": str(blend_mode), 
                "blend_factor": str(blend_factor) 
            }

        try:
            save_file(lora, lora_path, metadata=metadata)
            print(f"Quantized LoRA saved to: {lora_path}")
        except Exception as e:
            print(f"Error saving quantized LoRA: {e}")

        if metadata:
            lora["metadata"] = metadata

    def load_and_quantize_lora(self, lora_name, quantization_bits, strength_model, strength_clip,
                              model=None, clip=None, save_quantized_lora=False, save_name="quantized_lora",
                              quantization_iterations=1, stepwise_quantization=False, quantization_step_size=1,
                              blend_mode=False, blend_factor=0.5): 

        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)

        try:
            with safe_open(lora_path, framework="pt", device="cpu") as f:
                lora_metadata = f.metadata()
            lora = comfy.utils.load_torch_file(lora_path, safe_load=True)

        except Exception as e:
            print(f"Error loading LoRA: {e}")
            return (model, clip, None, None)

        original_dtype = None
        for key, tensor in lora.items():
            if "lora_down" in key or "lora_up" in key:
                if original_dtype is None:
                    original_dtype = tensor.dtype
                elif original_dtype == torch.float16 and tensor.dtype == torch.bfloat16:
                    original_dtype = torch.bfloat16

        original_bits = 32 if original_dtype == torch.float32 else 16
        quantized_lora = {}
        original_lora = lora.copy() 


        if quantization_bits >= original_bits:
            stepwise_quantization = False

        if stepwise_quantization:
             for key, tensor in lora.items():
                if key == "metadata":
                    quantized_lora[key] = tensor
                elif "lora_down" in key or "lora_up" in key:
                    temp_tensor = tensor
                    current_bits = original_bits
                    for _ in range(quantization_iterations):
                        while current_bits > quantization_bits:
                            current_bits = max(current_bits - quantization_step_size, quantization_bits)
                            temp_tensor = self._quantize_tensor(temp_tensor, current_bits)
                    quantized_lora[key] = temp_tensor
                else:
                    quantized_lora[key] = tensor
        else:
            for key, tensor in lora.items():
                if key == "metadata":
                    quantized_lora[key] = tensor
                elif "lora_down" in key or "lora_up" in key:
                    temp_tensor = tensor
                    for _ in range(quantization_iterations):
                        temp_tensor = self._quantize_tensor(temp_tensor, quantization_bits)
                    quantized_lora[key] = temp_tensor
                else:
                    quantized_lora[key] = tensor

        if blend_mode:
            blended_lora = {}
            for key in quantized_lora.keys():
                if key == "metadata":
                    blended_lora[key] = quantized_lora[key]
                elif "lora_down" in key or "lora_up" in key:
                    blended_lora[key] = (1 - blend_factor) * quantized_lora[key] + blend_factor * original_lora[key]
                else:
                    blended_lora[key] = quantized_lora[key]
            quantized_lora = blended_lora


        if model is not None and clip is not None:
            model_lora, clip_lora = comfy.sd.load_lora_for_models(model, clip, quantized_lora, strength_model, strength_clip)
        elif model is not None:
            model_lora, clip_lora = comfy.sd.load_lora_for_models(model, None, quantized_lora, strength_model, 0.0)
            clip = None
        elif clip is not None:
            model_lora, clip_lora = comfy.sd.load_lora_for_models(None, clip, quantized_lora, 0.0, strength_clip)
            model = None
        else:
            model_lora, clip_lora = None, None

        if save_quantized_lora:
            self._save_processed_lora(lora=quantized_lora, save_name=save_name, quantization_bits=quantization_bits,
                                      quantization_iterations=quantization_iterations, stepwise_quantization=stepwise_quantization,
                                      quantization_step_size=quantization_step_size,
                                      blend_mode=blend_mode, blend_factor=blend_factor) 

        lora_metadata_string = json.dumps(lora_metadata, indent=4) if lora_metadata else "{}"
        return (model_lora, clip_lora, quantized_lora, lora_metadata_string)