import time

import numpy as np

from knn.knn_predict import knn_generate, KNNPredictor
from preprocess.processor import get_processed_data, get_file_name
from os import path

from test_data import test_manager
from test_data.test_manager import results_accuracy
from util.file import load_or_create
import matplotlib.pyplot as plt

(new_X, new_y,
 max_t_s_num,
 num_student,
 test_new_X, test_new_y) = get_processed_data()

n = 2
print('Using n of {}'.format(n))

extra = '{}'.format(get_file_name())

file_name = 'knn_{}_{}'.format(extra, n)

knn_trained = load_or_create(file_name,
                             create_fn=lambda: knn_generate(new_X, new_y, n_neighbors=n, verbose=True),
                             folder=path.join('data', 'model', 'knn'))

# test_data.test(
#     predict_fn=
#     lambda face_encodings: predict(face_encodings, knn_trained,
#                                    distance_threshold=0.42,
#                                    n_neighbors=n),
#     show_image=False
# )

hist_acc = []
hist_tol = []
step = 0.01
start = time.time()
for tolerance in np.arange(0, 1 + step, step):
    test_result = test_manager.test_predictor(
        predictor=KNNPredictor(
            model_name=file_name,
            n=n,
            tolerance=tolerance,
            print_time=False
        ),
        show_image=False,
        print_info=False
    )
    accuracy = results_accuracy(test_result)
    print("At a tolerance of {:.3f}, accuracy is {:.2f}%".format(tolerance, accuracy * 100))
    hist_acc.append(accuracy)
    hist_tol.append(tolerance)
print("Test finished in {:.3f}ms".format((time.time() - start) * 1000))

plt.plot(hist_tol, hist_acc)
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('tolerance')
plt.show()
