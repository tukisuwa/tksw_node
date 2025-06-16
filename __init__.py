from .image_sequence_loader import ImageSequenceLoader
from .image_pair_sequence_loader import ImagePairSequenceLoader
from .text_combiner import TextCombiner
from .text_processor import TextProcessor
from .lora_loader_elemental import LoraLoaderElemental
from .random_word_replacer import RandomWordReplacer
from .lora_weight_randomizer import LoraWeightRandomizer
from .lora_mixer_elemental import LoraMixerElemental
from .quantized_lora_loader import QuantizedLoraLoader
from .lora_selector import LoraSelector
from .text_file_selector import TextFileSelector
from .Image_text_pair_sequence_loader import ImageTextPairSequenceLoader
from .custom_cfg_schedule import CustomCFGSchedule
from .image_storage_nodes import (
    StoreImageByNumber,
    RetrieveImageByNumber,
    StoreMultipleImagesByNumber,
    RetrieveMultipleImagesByNumber
)

NODE_CLASS_MAPPINGS = {
    "ImageSequenceLoader": ImageSequenceLoader,
    "ImagePairSequenceLoader": ImagePairSequenceLoader,
    "TextCombiner": TextCombiner,
    "TextProcessor": TextProcessor,
    "LoraLoaderElemental": LoraLoaderElemental,
    "RandomWordReplacer": RandomWordReplacer,
    "LoraWeightRandomizer": LoraWeightRandomizer,
    "LoraMixerElemental": LoraMixerElemental,
    "QuantizedLoraLoader": QuantizedLoraLoader,
    "LoraSelector": LoraSelector,
    "TextFileSelector": TextFileSelector,
    "ImageTextPairSequenceLoader": ImageTextPairSequenceLoader,
    "StoreImageByNumber": StoreImageByNumber,
    "RetrieveImageByNumber": RetrieveImageByNumber,
    "StoreMultipleImagesByNumber": StoreMultipleImagesByNumber,
    "RetrieveMultipleImagesByNumber": RetrieveMultipleImagesByNumber,
    "CustomCFGSchedule": CustomCFGSchedule,

}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageSequenceLoader": "Image Sequence Loader",
    "ImagePairSequenceLoader": "Image Pair Sequence Loader",
    "TextCombiner": "Text Combiner",
    "TextProcessor": "Text Processor",
    "LoraLoaderElemental": "Lora Loader Elemental",
    "RandomWordReplacer": "Random Word Replacer",
    "LoraWeightRandomizer": "Lora Weight Randomizer",
    "LoraMixerElemental": "Lora Mixer Elemental",
    "QuantizedLoraLoader": "Quantized Lora Loader",
    "LoraSelector": "Lora Selector",
    "TextFileSelector": "Text File Selector",
    "ImageTextPairSequenceLoader": "Image TextPair SequenceLoader",
    "StoreImageByNumber": "Store Image by Number",
    "RetrieveImageByNumber": "Retrieve Image by Number",
    "StoreMultipleImagesByNumber": "Store Multiple Images by Number",
    "RetrieveMultipleImagesByNumber": "Retrieve Multiple Images by Number",
    "CustomCFGSchedule": "Custom CFG Schedule",
}


__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']