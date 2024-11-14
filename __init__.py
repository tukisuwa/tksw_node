from .image_sequence_loader import ImageSequenceLoader
from .image_pair_sequence_loader import ImagePairSequenceLoader
from .text_combiner import TextCombiner
from .text_processor import TextProcessor


NODE_CLASS_MAPPINGS = {
    "ImageSequenceLoader": ImageSequenceLoader,
    "ImagePairSequenceLoader": ImagePairSequenceLoader,
    "TextCombiner": TextCombiner,
    "TextProcessor": TextProcessor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageSequenceLoader": "Image Sequence Loader",
    "ImagePairSequenceLoader": "Image Pair Sequence Loader",
    "TextCombiner": "Text Combiner",
    "TextProcessor": "Text Processor",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']