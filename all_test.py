import time

from knn.knn_predict import KNNPredictor
from knn_kmeans.knn_kmeans_predict import KNNKmeansPredictor
from nn.nn_predict import NNPredictor
from svm.svm_predict import SVMPredictor
from test_data import test_manager
from test_data.test_manager import results_stat
import numpy as np
import matplotlib.pyplot as plt

tolerance = 0.55

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
        convert_power=0.5,
        print_time=False
    ),
    # NN_FD_Predictor(
    #     model_name='nn_y_1000_64_tanh',
    #     tolerance=tolerance,
    #     print_time=False
    # )
]

hist_accs = [[] for _ in testing_predictors]
hist_fps = [[] for _ in testing_predictors]
hist_fns = [[] for _ in testing_predictors]
hist_tol = []
step = 0.005
start = time.time()
for tolerance in np.arange(0, 1 + step, step):
    for idx, predictor in enumerate(testing_predictors):
        tolerance_matters = hasattr(predictor, 'tolerance')
        if tolerance_matters or (len(hist_accs[idx]) == 0):
            if tolerance_matters:
                predictor.tolerance = tolerance
                if hasattr(predictor, 'update'):
                    predictor.update(['tolerance'])
            test_result = test_manager.test_predictor(
                predictor=predictor,
                show_image=False,
                print_info=False
            )
            stat = results_stat(test_result)
            hist_accs[idx].append(stat['accuracy'])
            hist_fps[idx].append(stat['false_positive_rate'])
            hist_fns[idx].append(stat['false_negative_rate'])
        else:
            hist_accs[idx].append(hist_accs[idx][0])
            hist_fps[idx].append(hist_fps[idx][0])
            hist_fns[idx].append(hist_fns[idx][0])

    hist_tol.append(tolerance)


def my_plot(hist_blas, tols, bla, xname='tolerance', legend=None):
    plt.rcParams["figure.figsize"] = (25, 15)
    ax = plt.figure().gca()
    ax.set_xticks(np.arange(0, 1 + step, 0.025))
    ax.set_yticks(np.arange(0, 1 + step, 0.025))
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


my_plot(hist_accs, hist_tol, 'accuracy')
my_plot(hist_fps, hist_tol, 'false positive')
my_plot(hist_fns, hist_tol, 'false negative')

plot_arr = []
for i in plot_arr:
    my_plot([hist_accs[i], hist_fps[i], hist_fns[i]], hist_tol,
            '{} model'.format(testing_predictors[i].__class__.__name__.replace('Predictor', '')),
            legend=['accuracy', 'false positive', 'false negative'])
