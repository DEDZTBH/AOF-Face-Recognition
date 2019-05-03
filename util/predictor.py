class EncodingsPredictor:
    def __init__(self, **kwargs):
        raise NotImplementedError()

    def predict(self, face_encodings):
        raise NotImplementedError()


def get_param(name, kwas):
    if name in kwas:
        return kwas[name]
    else:
        raise AttributeError('Need parameter {}'.format(name))


def get_param_default(name, default, kwas):
    return kwas[name] if name in kwas else default
