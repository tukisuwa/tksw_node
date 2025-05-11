import torch
from typing import Tuple, Dict, Any, List, Optional

shared_image_pool: Dict[int, torch.Tensor] = {}

MAX_IMAGE_SLOTS = 5

class StoreImageByNumber:

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "image": ("IMAGE",),
                "image_id": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "step": 1}),
                "skip_if_exists": ("BOOLEAN", {"default": False, "label_on": "Skip if ID exists", "label_off": "Overwrite if ID exists"}),
            }
        }

    RETURN_TYPES: Tuple[()] = ()
    FUNCTION: str = "store_image"
    OUTPUT_NODE: bool = True
    CATEGORY = "tksw_node"

    @classmethod
    def IS_CHANGED(cls, image: torch.Tensor, image_id: int, skip_if_exists: bool) -> float:
        return float("NaN")

    def store_image(self, image: torch.Tensor, image_id: int, skip_if_exists: bool) -> Dict[str, Any]:
        log_prefix = f"[Store Image (Memory)] Number ID {image_id}"

        if skip_if_exists and image_id in shared_image_pool:
            message = f"{log_prefix}: ID already exists and 'skip_if_exists' is True. Skipped."
            print(message)
            return {"ui": {"text": f"ID {image_id} skipped."}}

        is_overwrite = image_id in shared_image_pool
        shared_image_pool[image_id] = image.clone()
        
        if skip_if_exists: 
            action_msg = "newly stored"
            print(f"{log_prefix}: Image {action_msg}. Shape: {image.shape}, Device: {image.device} (skip_if_exists: True)")
        else:
            action_msg = "overwritten" if is_overwrite else "newly stored"
            print(f"{log_prefix}: Image {action_msg}. Shape: {image.shape}, Device: {image.device} (skip_if_exists: False)")
            
        return {"ui": {"text": f"ID {image_id} {action_msg}."}}

class RetrieveImageByNumber:

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "image_id": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "step": 1}),
                "remove_after_retrieval": ("BOOLEAN", {"default": False, "label_on": "Remove after retrieval", "label_off": "Keep after retrieval"}),
            },
            "optional": {
                "fallback_image": ("IMAGE",)
            }
        }

    RETURN_TYPES: Tuple[str, ...] = ("IMAGE",)
    RETURN_NAMES: Tuple[str, ...] = ("image",)
    FUNCTION: str = "retrieve_image"
    CATEGORY = "tksw_node"

    @classmethod
    def IS_CHANGED(cls, image_id: int, remove_after_retrieval: bool, fallback_image: Optional[torch.Tensor] = None) -> float:
        return float("NaN")

    def retrieve_image(self, image_id: int, remove_after_retrieval: bool, fallback_image: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor]:
        retrieved_image: Optional[torch.Tensor] = None
        log_prefix = f"[Retrieve Image (Memory)] Number ID {image_id}"

        if image_id in shared_image_pool:
            retrieved_image = shared_image_pool[image_id].clone()
            print(f"{log_prefix}: Image retrieved. Shape: {retrieved_image.shape}, Device: {retrieved_image.device}")
            
            if remove_after_retrieval:
                del shared_image_pool[image_id]
                print(f"{log_prefix}: Image removed from pool (remove_after_retrieval: True).")
            else:
                print(f"{log_prefix}: Image kept in pool (remove_after_retrieval: False).")
        else:
            print(f"{log_prefix}: Image not found in memory.")

        if retrieved_image is not None:
            return (retrieved_image,)

        if fallback_image is not None:
            print(f"{log_prefix}: Using provided fallback image.")
            return (fallback_image.clone(),)
        else:
            print(f"{log_prefix}: No fallback image provided. Outputting default 64x64 black image.")
            default_img = torch.zeros((1, 64, 64, 3), dtype=torch.float32, device="cpu")
            return (default_img,)

class StoreMultipleImagesByNumber:
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        inputs: Dict[str, Any] = {
            "required": {
                "skip_if_exists": ("BOOLEAN", {"default": False, "label_on": "Skip if ID exists", "label_off": "Overwrite if ID exists"}),
            },
            "optional": {}
        }
        for i in range(1, MAX_IMAGE_SLOTS + 1):
            inputs["optional"][f"image_{i}"] = ("IMAGE",)
            inputs["optional"][f"image_id_{i}"] = ("INT", {"default": i - 1, "min": 0, "max": 0xffffffffffffffff, "step": 1})
        return inputs

    RETURN_TYPES: Tuple[()] = ()
    FUNCTION: str = "store_images"
    OUTPUT_NODE: bool = True
    CATEGORY = "tksw_node"

    @classmethod
    def IS_CHANGED(cls, **kwargs: Any) -> float:
        return float("NaN")

    def store_images(self, skip_if_exists: bool, **kwargs: Any) -> Dict[str, Any]:
        stored_count = 0
        skipped_count = 0
        processed_ids_info: List[str] = []
        
        for i in range(1, MAX_IMAGE_SLOTS + 1):
            image_slot_key = f"image_{i}"
            image_id_slot_key = f"image_id_{i}"

            image: Optional[torch.Tensor] = kwargs.get(image_slot_key)
            image_id: Optional[int] = kwargs.get(image_id_slot_key) 

            if image is not None and image_id is not None: 
                log_prefix = f"[Store Multiple (Memory)] Slot {i} (ID {image_id})"
                
                if skip_if_exists and image_id in shared_image_pool:
                    print(f"{log_prefix}: ID exists, 'skip_if_exists' is True. Skipped.")
                    skipped_count += 1
                    processed_ids_info.append(f"ID {image_id}(S{i}):skipped")
                    continue
                
                is_overwrite = image_id in shared_image_pool
                shared_image_pool[image_id] = image.clone()
                stored_count += 1
                
                if skip_if_exists: 
                    action_msg = "newly stored"
                    processed_ids_info.append(f"ID {image_id}(S{i}):stored")
                else: 
                    action_msg = "overwritten" if is_overwrite else "newly stored"
                    processed_ids_info.append(f"ID {image_id}(S{i}):{'overwritten' if is_overwrite else 'stored'}")

                print(f"{log_prefix}: Image {action_msg}. Shape: {image.shape} (skip_if_exists: {skip_if_exists})")
            elif image is None and kwargs.get(image_id_slot_key) is not None:
                 print(f"[Store Multiple (Memory)] Slot {i} (ID {image_id}): image_id provided but no image. Skipped.")

        ui_summary = f"Stored: {stored_count}, Skipped: {skipped_count}."
        if processed_ids_info:
             ui_summary += " Details: " + ", ".join(processed_ids_info)
        if stored_count == 0 and skipped_count == 0:
            ui_summary = "No images/IDs provided to process."
            
        return {"ui": {"text": ui_summary}}


class RetrieveMultipleImagesByNumber:
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        inputs: Dict[str, Any] = {
            "required": {
                "remove_after_retrieval": ("BOOLEAN", {"default": False, "label_on": "Remove after retrieval", "label_off": "Keep after retrieval"}),
            },
            "optional": {
                "fallback_image": ("IMAGE",) 
            }
        }
        for i in range(1, MAX_IMAGE_SLOTS + 1):
            inputs["optional"][f"image_id_{i}"] = ("INT", {"default": i - 1, "min": 0, "max": 0xffffffffffffffff, "step": 1})
        return inputs

    RETURN_TYPES = tuple(["IMAGE"] * MAX_IMAGE_SLOTS)
    RETURN_NAMES = tuple([f"image_{i}" for i in range(1, MAX_IMAGE_SLOTS + 1)])
    FUNCTION = "retrieve_images"
    CATEGORY = "tksw_node"

    @classmethod
    def IS_CHANGED(cls, **kwargs: Any) -> float:
        return float("NaN")

    def retrieve_images(self, remove_after_retrieval: bool, **kwargs: Any) -> Tuple[torch.Tensor, ...]:
        outputs: List[torch.Tensor] = []
        node_fallback_image: Optional[torch.Tensor] = kwargs.get("fallback_image")

        for i in range(1, MAX_IMAGE_SLOTS + 1):
            image_id_slot_key = f"image_id_{i}"
            image_id: int = kwargs[image_id_slot_key] 

            retrieved_image: Optional[torch.Tensor] = None
            log_prefix = f"[Retrieve Multiple (Memory)] Slot {i} (ID {image_id})"
            
            if image_id in shared_image_pool:
                retrieved_image = shared_image_pool[image_id].clone()
                print(f"{log_prefix}: Image retrieved. Shape: {retrieved_image.shape}")
                
                if remove_after_retrieval:
                    del shared_image_pool[image_id]
                    print(f"{log_prefix}: Image removed from pool.")
                else:
                    print(f"{log_prefix}: Image kept in pool.")
            else:
                print(f"{log_prefix}: Image not found in memory.")

            if retrieved_image is None: 
                if node_fallback_image is not None:
                    retrieved_image = node_fallback_image.clone() 
                    print(f"{log_prefix}: Using node fallback image.")
                else:
                    default_img = torch.zeros((1, 64, 64, 3), dtype=torch.float32, device="cpu")
                    retrieved_image = default_img
                    print(f"{log_prefix}: No node fallback. Using default black image.")
            
            outputs.append(retrieved_image)
        
        return tuple(outputs)
