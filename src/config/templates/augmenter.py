from config import Config


class Augmenter(Config):
    pass


class ImageAugmenter(Augmenter):
    pass


class ColorImageAugmenter(ImageAugmenter):
    pass


class GrayImageAugmenter(ImageAugmenter):
    pass


class VectorAugmenter(Augmenter):
    gauss_std: float = 0.1
    dropout_prob: float = 0.25
