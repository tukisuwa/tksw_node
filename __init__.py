from .image_sequence_loader import ImageSequenceLoader
from .image_pair_sequence_loader import ImagePairSequenceLoader

NODE_CLASS_MAPPINGS = {
    "ImageSequenceLoader": ImageSequenceLoader,
    "ImagePairSequenceLoader": ImagePairSequenceLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageSequenceLoader": "Image Sequence Loader",
    "ImagePairSequenceLoader": "Image Pair Sequence Loader",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']