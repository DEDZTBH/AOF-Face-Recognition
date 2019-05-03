import time
from os import path

import numpy as np
from sklearn import svm

from util.file import load
from util.predictor import EncodingsPredictor, get_param, get_param_default


def train_svm(new_X_num, new_y_num, num_student):
    start = time.time()
    X = np.array(new_X_num)
    total_t_s_num = len(new_y_num)
    total_t_s_num_sqrt = np.sqrt(total_t_s_num)
    svms = []
    for name_id in range(num_student):
        y = np.array([int(num == name_id) for num in new_y_num])
        clf = svm.LinearSVC(class_weight={
            1: total_t_s_num_sqrt
        }, max_iter=5000)
        clf.fit(X, y)
        svms.append(clf)
    print('Trained SVM model in {:.3f}ms'.format((time.time() - start) * 1000))
    return svms


def predict(arr_face, svms, num_map, print_time=False):
    if print_time:
        start = time.time()
    processed_results = np.array([svm_i.predict(arr_face) for svm_i in svms]).transpose()
    result_names = []
    for result in processed_results:
        valid_idx = np.where(result == 1)[0]
        if len(valid_idx) == 0:
            # print('0!')
            result_names.append('Unknown')
        elif len(valid_idx) == 1:
            # print('1!')
            result_names.append(num_map[valid_idx[0]])
        else:
            print('Magic! {}'.format(len(valid_idx)))
            result_names.append(num_map[valid_idx[0]])
    if print_time:
        print('Made {} predictions in {:.3f}ms'.format(len(arr_face), (time.time() - start) * 1000))
    return result_names


class SVMPredictor(EncodingsPredictor):
    def __init__(self, **kwargs):
        (self.svms, self.num_map) = load(
            filename=get_param('model_name', kwargs),
            folder=path.join('data', 'model', 'svm'))
        self.print_time = get_param_default('print_time', False, kwargs)

    def predict(self, face_encodings):
        return predict(arr_face=face_encodings,
                       svms=self.svms,
                       num_map=self.num_map,
                       print_time=self.print_time)
