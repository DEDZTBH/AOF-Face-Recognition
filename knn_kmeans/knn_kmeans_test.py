import time
from os import path

from knn_kmeans.knn_kmeans_predict import knn_kmeans_generate, KNNKmeansPredictor
from preprocess.processor_num_map import get_processed_data, get_file_name
from test_data import test_manager
from test_data.test_manager import results_accuracy

import numpy as np

from util.file import load_or_create
import matplotlib.pyplot as plt

restore = False

(new_X_num, num_map, new_y_num,
 max_t_s_num,
 num_student,
 test_new_X_num, test_new_y) = get_processed_data()

n = 2
print('Using n of {}'.format(n))

extra = '{}'.format(get_file_name())

file_name = 'knn_{}_{}'.format(extra, n)

knn_trained = load_or_create(file_name,
                             create_fn=lambda: knn_kmeans_generate(restore, n, num_map),
                             folder=path.join('data', 'model', 'knn'))

# test_data.test(
#     predict_fn=
#     lambda face_encodings: predict(face_encodings, knn_trained,
#                                    distance_threshold=0.44,
#                                    n_neighbors=n),
#     show_image=False
# )

hist_acc = []
hist_tol = []
step = 0.01
start = time.time()
for tolerance in np.arange(0, 1 + step, step):
    test_result = test_manager.test_predictor(
        predictor=
        KNNKmeansPredictor(
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
