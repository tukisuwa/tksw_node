from .image_sequence_loader import ImageSequenceLoader
from .image_pair_sequence_loader import ImagePairSequenceLoader
from .text_combiner import TextCombiner
from .text_processor import TextProcessor
from .lora_loader_elemental import LoraLoaderElemental
from .random_word_replacer import RandomWordReplacer
from .lora_weight_randomizer import LoraWeightRandomizer


NODE_CLASS_MAPPINGS = {
    "ImageSequenceLoader": ImageSequenceLoader,
    "ImagePairSequenceLoader": ImagePairSequenceLoader,
    "TextCombiner": TextCombiner,
    "TextProcessor": TextProcessor,
    "LoraLoaderElemental": LoraLoaderElemental,
    "RandomWordReplacer": RandomWordReplacer,
    "LoraWeightRandomizer": LoraWeightRandomizer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageSequenceLoader": "Image Sequence Loader",
    "ImagePairSequenceLoader": "Image Pair Sequence Loader",
    "TextCombiner": "Text Combiner",
    "TextProcessor": "Text Processor",
    "LoraLoaderElemental": "Lora Loader Elemental",
    "RandomWordReplacer": "Random Word Replacer",
    "LoraWeightRandomizer": "Lora Weight Randomizer",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']