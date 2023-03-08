MODEL_CLASSES = {}
LOSS_TERM_CLASSES = {}


def _register(name, obj, dct):
    dct[name] = obj


def register_model(cls):
    _register(cls.__name__, cls, MODEL_CLASSES)
    return cls


def register_loss_term(cls):
    _register(cls.__name__, cls, LOSS_TERM_CLASSES)
    return cls

