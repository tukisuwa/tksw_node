from folder_paths import get_filename_list
from nodes import LoraLoader
import random
import torch

LORA_COUNT = 20

class LoraWeightRandomizer:
    def __init__(self):
        self.loras = [LoraLoader() for _ in range(LORA_COUNT)]

    @classmethod
    def INPUT_TYPES(cls):
        args = {
            "model": ("MODEL",),
            "clip": ("CLIP",),
            "total_strength": ("FLOAT", {"default": 1.00, "min": 0.00, "max": 10.00, "step": 0.01}),
            "max_single_strength": ("FLOAT", {"default": 1.00, "min": 0.00, "max": 2.00, "step": 0.01}),
            "randomize_total_strength": ("BOOLEAN", {"default": False}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
        }
        arg_lora_name = ([""] + get_filename_list("loras"),)
        for i in range(LORA_COUNT):
            args["{}:lora".format(i)] = arg_lora_name
        return {"required": args}

    def apply(self, model, clip, total_strength, max_single_strength, randomize_total_strength, seed, **kwargs):
        selected_loras = []
        for i in range(LORA_COUNT):
            lora_name = kwargs["{}:lora".format(i)]
            if lora_name != "":
                selected_loras.append((lora_name, i))

        if not selected_loras:
            return (model, clip, "") 

        torch.manual_seed(seed)
        random.seed(seed)

        if randomize_total_strength:
            total_strength = round(random.uniform(0, total_strength), 2)

        num_selected = len(selected_loras)
        strengths = [0.00] * num_selected
        remaining_strength = total_strength

        allocation_order = list(range(num_selected))
        random.shuffle(allocation_order)

        for i in allocation_order[:-1]:
            max_allowed = min(remaining_strength, max_single_strength)
            strength = round(random.uniform(0, max_allowed), 2)
            strengths[i] = strength
            remaining_strength -= strength

        strengths[allocation_order[-1]] = round(min(remaining_strength, max_single_strength), 2)

        allocated_strength_total = sum(strengths)
        if allocated_strength_total < total_strength:
            diff = total_strength - allocated_strength_total
            num_under_max = sum(1 for s in strengths if s < max_single_strength)
            if num_under_max > 0:
                increment = diff / num_under_max
                for i in range(len(strengths)):
                    if strengths[i] < max_single_strength:
                        add = min(increment, max_single_strength - strengths[i])
                        strengths[i] = round(strengths[i] + add, 2)

        reordered_strengths = [0.00] * num_selected
        for i, original_index in enumerate(allocation_order):
            reordered_strengths[original_index] = strengths[i]
        strengths = reordered_strengths

        output_text = f"LoraWeightRandomizer Settings:\n"
        output_text += f"  Total Strength: {total_strength:.2f}\n"
        output_text += f"  Max Single Strength: {max_single_strength:.2f}\n"
        output_text += f"  Randomize Total Strength: {randomize_total_strength}\n"
        output_text += f"  Seed: {seed}\n"
        output_text += "LoRA Weights:\n"

        for (lora_name, index), strength in zip(selected_loras, strengths):
            output_text += f"  - {lora_name}: {strength:.2f}\n"
            (model, clip) = self.loras[index].load_lora(model, clip, lora_name, strength, strength)


        return (model, clip, output_text)


    RETURN_TYPES = ("MODEL", "CLIP", "STRING")
    RETURN_NAMES = ("MODEL", "CLIP", "settings")
    FUNCTION = "apply"
    CATEGORY = "tksw_node"