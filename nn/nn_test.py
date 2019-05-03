import pickle
import time

import numpy as np
from tensorflow.python.keras.models import load_model

from nn.nn_predict import train_nn, get_file_name, get_file_path, NNPredictor
from preprocess.processor_num_map import get_processed_data
from test_data import test_manager
from test_data.test_manager import results_accuracy

import matplotlib.pyplot as plt

(new_X_num, num_map, new_y_num,
 max_t_s_num,
 num_student,
 orig_new_X_num, orig_new_y) = get_processed_data()

try:
    fp = get_file_path()
    recog_model = load_model(fp)
    with open('{}.num_map.pkl'.format(fp), 'rb') as file:
        num_map = pickle.load(file)
except OSError:
    recog_model, num_map = train_nn(new_X_num, new_y_num, num_student, num_map, True)

# test_result = test_manager.test(
#     predict_fn=lambda arr_face:
#     predict(arr_face, recog_model, 0.52, True),
#     show_image=True
# )
# accuracy = results_accuracy(test_result)
# print("Accuracy is {:.2f}%".format(accuracy * 100))

hist_acc = []
hist_tol = []
step = 0.005
start = time.time()
i_NNPredictor = NNPredictor(
    model_name=get_file_name(),
    tolerance=0,
    print_time=False
)
for tolerance in np.arange(0, 1 + step, step):
    i_NNPredictor.tolerance = tolerance
    test_result = test_manager.test_predictor(
        predictor=i_NNPredictor,
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
