import time

from knn.knn_predict import KNNPredictor
from knn_kmeans.knn_kmeans_predict import KNNKmeansPredictor
from naive.naive_predict import NaivePredictor
from nn.nn_predict import NNPredictor
from preprocess.processor import get_processed_data
from svm.svm_predict import SVMPredictor
from test_data import test_manager
from test_data.test_manager import results_accuracy
import numpy as np
import matplotlib.pyplot as plt

(new_X, new_y,
 max_t_s_num,
 num_student,
 orig_new_X, orig_new_y) = get_processed_data()

tolerance = 0

testing_predictors = [
    NaivePredictor(
        known_encodings=new_X,
        names=new_y,
        tolerance=tolerance,
        print_time=False
    ),
    KNNPredictor(
        model_name='knn_preprocess_y_0_100_neq_2',
        n=2,
        tolerance=tolerance,
        print_time=False
    ),
    KNNKmeansPredictor(
        model_name='knn_preprocess_num_map_y_0_100_neq_2',
        n=2,
        tolerance=tolerance,
        print_time=False
    ),
    SVMPredictor(
        model_name='svm_y'
    ),
    NNPredictor(
        model_name='nn_y_1000_64_tanh',
        tolerance=tolerance,
        print_time=False
    )
]

hist_accs = [[] for _ in testing_predictors]
hist_tol = []
step = 0.001
start = time.time()
for tolerance in np.arange(0, 1 + step, step):
    for idx, predictor in enumerate(testing_predictors):
        tolerance_matters = hasattr(predictor, 'tolerance')
        if tolerance_matters or (len(hist_accs[idx]) == 0):
            if tolerance_matters:
                predictor.tolerance = tolerance
            test_result = test_manager.test_predictor(
                predictor=predictor,
                show_image=False,
                print_info=False
            )
            accuracy = results_accuracy(test_result)
            hist_accs[idx].append(accuracy)
        else:
            hist_accs[idx].append(hist_accs[idx][0])
    hist_tol.append(tolerance)


ax = plt.figure().gca()
ax.set_xticks(np.arange(0, 1.001, 0.05))
ax.set_yticks(np.arange(0, 1.001, 0.05))
for hist_acc in hist_accs:
    magic = 1.5 + np.random.rand() * 3
    plt.plot(hist_tol, hist_acc,
             linestyle='dashed',
             dashes=(magic, magic))
plt.grid()
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('tolerance')
plt.legend([predictor.__class__.__name__.replace('Predictor', '') for predictor in testing_predictors],
           loc='upper left')
plt.show(figsize=(2520, 1080))
