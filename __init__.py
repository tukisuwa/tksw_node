from .image_sequence_loader import ImageSequenceLoader
from .image_pair_sequence_loader import ImagePairSequenceLoader
from .Image_text_pair_sequence_loader import ImageTextPairSequenceLoader
from .ImageLoaderSeedSync import ImageLoaderSeedSync
from .text_file_selector import TextFileSelector
from .text_combiner import TextCombiner
from .text_processor import TextProcessor
from .random_word_replacer import RandomWordReplacer
from .advanced_fast_preview import AdvancedFastPreview, SimpleFastPreview
from .text_fast_preview import TextFastPreview
from .lora_loader_elemental import LoraLoaderElemental, LoraLoaderElementalUI
from .lora_weight_randomizer import LoraWeightRandomizer
from .lora_mixer_elemental import LoraMixerElemental
from .quantized_lora_loader import QuantizedLoraLoader
from .lora_selector import LoraSelector
from .image_storage_nodes import (
    StoreImageByNumber,
    RetrieveImageByNumber,
    StoreMultipleImagesByNumber,
    RetrieveMultipleImagesByNumber
)
from .custom_cfg_schedule import CustomCFGSchedule

WEB_DIRECTORY = "js"

NODE_CLASS_MAPPINGS = {
    # Data loading
    "ImageSequenceLoader": ImageSequenceLoader,
    "ImagePairSequenceLoader": ImagePairSequenceLoader,
    "ImageTextPairSequenceLoader": ImageTextPairSequenceLoader,
    "ImageLoaderSeedSync": ImageLoaderSeedSync,
    "TextFileSelector": TextFileSelector,

    # Text processing
    "TextCombiner": TextCombiner,
    "TextProcessor": TextProcessor,
    "RandomWordReplacer": RandomWordReplacer,

    # Preview
    "SimpleFastPreview": SimpleFastPreview,
    "AdvancedFastPreview": AdvancedFastPreview,
    "TextFastPreview": TextFastPreview,

    # LoRA
    "LoraLoaderElemental": LoraLoaderElemental,
    "LoraLoaderElementalUI": LoraLoaderElementalUI,
    "LoraWeightRandomizer": LoraWeightRandomizer,
    "LoraMixerElemental": LoraMixerElemental,
    "QuantizedLoraLoader": QuantizedLoraLoader,
    "LoraSelector": LoraSelector,

    # Image utilities
    "StoreImageByNumber": StoreImageByNumber,
    "RetrieveImageByNumber": RetrieveImageByNumber,
    "StoreMultipleImagesByNumber": StoreMultipleImagesByNumber,
    "RetrieveMultipleImagesByNumber": RetrieveMultipleImagesByNumber,

    # Sampling
    "CustomCFGSchedule": CustomCFGSchedule,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # Data loading
    "ImageSequenceLoader": "Image Sequence Loader",
    "ImagePairSequenceLoader": "Image Pair Sequence Loader",
    "ImageTextPairSequenceLoader": "Image TextPair SequenceLoader",
    "ImageLoaderSeedSync": "Image Loader Seed Sync",
    "TextFileSelector": "Text File Selector",

    # Text processing
    "TextCombiner": "Text Combiner",
    "TextProcessor": "Text Processor",
    "RandomWordReplacer": "Random Word Replacer",

    # Preview
    "SimpleFastPreview": "Simple Fast Preview",
    "AdvancedFastPreview": "Advanced Fast Preview",
    "TextFastPreview": "Text Fast Preview",

    # LoRA
    "LoraLoaderElemental": "LoRA Loader Elemental",
    "LoraLoaderElementalUI": "LoRA Loader Elemental UI",
    "LoraWeightRandomizer": "LoRA Weight Randomizer",
    "LoraMixerElemental": "LoRA Mixer Elemental",
    "QuantizedLoraLoader": "Quantized LoRA Loader",
    "LoraSelector": "LoRA Selector",

    # Image utilities
    "StoreImageByNumber": "Store Image by Number",
    "RetrieveImageByNumber": "Retrieve Image by Number",
    "StoreMultipleImagesByNumber": "Store Multiple Images by Number",
    "RetrieveMultipleImagesByNumber": "Retrieve Multiple Images by Number",

    # Sampling
    "CustomCFGSchedule": "Custom CFG Schedule",
}


__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
