import time
import numpy as np
import face_recognition

from util.predictor import EncodingsPredictor, get_param, get_param_default


def predict(face_encodings, known_encodings, names, tolerance=0.54, print_time=False):
    start = time.time()

    def predict_single(face_encoding):
        # See if the face is a match for the known face(s)
        face_dis = face_recognition.face_distance(known_encodings, face_encoding)

        name = 'Unknown'

        best_match_index = np.argmin(face_dis)
        if face_dis[best_match_index] <= tolerance:
            name = names[best_match_index]
        return name

    results = [predict_single(x) for x in face_encodings]
    if print_time:
        print('Made {} predictions in {:.3f}ms'.format(len(face_encodings), (time.time() - start) * 1000))

    return results


class NaivePredictor(EncodingsPredictor):
    def __init__(self, **kwargs):
        self.known_encodings = get_param('known_encodings', kwargs)
        self.names = get_param('names', kwargs)
        self.tolerance = get_param_default('tolerance', 0.54, kwargs)
        self.print_time = get_param_default('print_time', False, kwargs)

    def predict(self, face_encodings):
        return predict(face_encodings=face_encodings,
                       known_encodings=self.known_encodings,
                       names=self.names,
                       tolerance=self.tolerance,
                       print_time=self.print_time)
