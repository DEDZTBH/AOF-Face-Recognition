import time

import numpy as np

from src.knn import knn_generate, predict
from src.preprocess.processor import get_processed_data, get_file_name
from os import path

from src.test_data import test_data
from src.test_data.test_data import results_accuracy
from src.util.util import load_or_create

(new_X, new_X_raw, new_y,
 max_t_s_num,
 num_student,
 test_new_X, test_new_y) = get_processed_data()

n = round(max_t_s_num * 0.5)
print('Using n of {}'.format(n))

extra = '1719_{}'.format(get_file_name())

knn_trained = load_or_create('knn_{}_{}'.format(extra, n),
                             create_fn=lambda: knn_generate(new_X, new_y, n_neighbors=n, verbose=True),
                             folder=path.join('pkl', 'knn'))

# test_data.test(
#     predict_fn=
#     lambda face_encodings: predict(face_encodings, knn_trained,
#                                    distance_threshold=0.42,
#                                    n_neighbors=n),
#     show_image=False
# )

start = time.time()
for tolerance in np.arange(0.40, 0.61, 0.01):
    test_result = test_data.test(
        predict_fn=
        lambda face_encodings: predict(face_encodings, knn_trained,
                                       distance_threshold=tolerance,
                                       n_neighbors=n, print_time=False),
        show_image=False,
        print_info=False
    )
    accuracy = results_accuracy(test_result)
    print("At a tolerance of {:.2f}, accuracy is {:.2f}%".format(tolerance, accuracy * 100))
print("Test finished in {:.3f}ms".format((time.time() - start) * 1000))
