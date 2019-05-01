import time

import numpy as np
from sklearn import svm

from preprocess.processor_num_map import get_processed_data
from test_data import test_manager

(new_X_num, num_map, new_y_num,
 max_t_s_num,
 num_student,
 orig_new_X_num, orig_new_y) = get_processed_data()


def train():
    start = time.time()
    X = np.array(new_X_num)
    total_t_s_num = len(new_y_num)
    total_t_s_num_sqrt = np.sqrt(total_t_s_num)
    svms = []
    for name_id in range(num_student):
        y = np.array([int(num == name_id) for num in new_y_num])
        clf = svm.LinearSVC(class_weight={
            1: total_t_s_num_sqrt
        })
        clf.fit(X, y)
        svms.append(clf)
    print('Trained SVM model in {:.3f}ms'.format((time.time() - start) * 1000))
    return svms


def predict(arr_face, svms, print_time=False):
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


if __name__ == '__main__':
    svms = train()
    arr_face = [new_X_num[0], new_X_num[2]]
    test_manager.test(
        predict_fn=lambda arr_face:
        predict(arr_face, svms, True),
        show_image=False
    )
