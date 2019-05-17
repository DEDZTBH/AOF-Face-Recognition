import time
from knn.knn_predict import KNNPredictor
from knn_kmeans.knn_kmeans_predict import KNNKmeansPredictor
from naive.naive_predict import NaivePredictor
from nn.nn_predict import NNPredictor
from preprocess.processor import get_processed_data
from svm.svm_predict import SVMPredictor
from test_data import test_manager
import numpy as np
import matplotlib.pyplot as plt

trial_times = 5
tolerance = 0.55
prepare_times = 1

# (new_X, new_y,
#  max_t_s_num,
#  num_student,
#  orig_new_X, orig_new_y) = get_processed_data()


def start():
    global current_time
    current_time = time.time()


def stop(print_info=False):
    global current_time
    elapse = time.time() - current_time
    if print_info:
        print('{:2f}ms'.format(elapse * 1000))
    return elapse


testing_predictors = [
    # NaivePredictor(
    #     known_encodings=new_X,
    #     names=new_y,
    #     tolerance=tolerance,
    #     print_time=False
    # ),
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
        model_name='nn_y_500_64_tanh',
        tolerance=tolerance,
        print_time=False
    )
]

step = 10
num_encodings = np.arange(1, 400 + step, step)
time_results_s = [[0 for _ in num_encodings] for _ in testing_predictors]

# prepare
for i in range(prepare_times):
    encodings_to_test = test_manager.randomly_pick_encodings(num_encodings)
    len_encodings_to_test = len(encodings_to_test)
    print('Test {}/{}'.format(i+1, prepare_times))
    for e_i, encodings in enumerate(encodings_to_test):
        for p_i, predictor in enumerate(testing_predictors):
            predictor.predict(encodings)
            if p_i == 0:
                print('Encodings: {}/{}'.format(e_i+1, len_encodings_to_test))

for trial_time in range(trial_times):
    encodings_to_test = test_manager.randomly_pick_encodings(num_encodings)
    len_encodings_to_test = len(encodings_to_test)
    print('Trial {}/{}'.format(trial_time+1, trial_times))
    for e_i, encodings in enumerate(encodings_to_test):
        for p_i, predictor in enumerate(testing_predictors):
            start()
            predictor.predict(encodings)
            elapse = stop()
            time_results_s[p_i][e_i] += elapse
            if p_i == 0:
                print('Encodings: {}/{}'.format(e_i+1, len_encodings_to_test))

time_results_s = np.array(time_results_s) / trial_times * 1000


def my_plot(hist_blas, tols, bla, xname='people', legend=None):
    # plt.rcParams["figure.figsize"] = (25, 15)
    ax = plt.gca()
    ax.set_yscale('log')
    for hist_bla in hist_blas:
        magic = 1.5 + np.random.rand() * 3
        plt.plot(tols, hist_bla,
                 linestyle='dashed',
                 dashes=(magic, magic))
    plt.grid()
    plt.title('model {}'.format(bla))
    plt.ylabel(bla)
    plt.xlabel(xname)
    if legend:
        plt.legend(legend, loc='upper left')
    else:
        plt.legend([predictor.__class__.__name__.replace('Predictor', '') for predictor in testing_predictors],
                   loc='upper left')
    plt.show()


my_plot(time_results_s, num_encodings, 'time(ms) on 1070ti')
