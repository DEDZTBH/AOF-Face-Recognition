import time

from naive.naive_predict import NaivePredictor
from preprocess.processor import get_processed_data
import numpy as np

from test_data import test_manager
from test_data.test_manager import results_accuracy
import matplotlib.pyplot as plt

(new_X, new_y,
 max_t_s_num,
 num_student,
 orig_new_X, orig_new_y) = get_processed_data()

hist_acc = []
hist_tol = []
step = 0.01
start = time.time()
for tolerance in np.arange(0, 1 + step, step):
    test_result = test_manager.test_predictor(
        predictor=NaivePredictor(
            known_encodings=new_X,
            names=new_y,
            tolerance=tolerance,
            print_time=False
        ),
        show_image=False,
        print_info=False
    )
    accuracy = results_accuracy(test_result)
    hist_acc.append(accuracy)
    hist_tol.append(tolerance)
    print("At a tolerance of {:.3f}, accuracy is {:.2f}%".format(tolerance, accuracy * 100))
print("Test finished in {:.3f}ms".format((time.time() - start) * 1000))

plt.plot(hist_tol, hist_acc)
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('tolerance')
plt.show()
