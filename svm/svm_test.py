from os import path

from preprocess.processor_num_map import get_processed_data
from svm.svm_predict import train_svm, predict, SVMPredictor
from test_data import test_manager
from test_data.test_manager import results_accuracy
from util.file import load_or_create

(new_X_num, num_map, new_y_num,
 max_t_s_num,
 num_student,
 orig_new_X_num, orig_new_y) = get_processed_data()

file_name = 'svm_y'

(svms, num_map) = load_or_create(file_name,
                                 create_fn=lambda:
                                 (
                                     train_svm(new_X_num=new_X_num, new_y_num=new_y_num, num_student=num_student),
                                     num_map),
                                 folder=path.join('data', 'model', 'svm'))
test_result = test_manager.test_predictor(
    predictor=SVMPredictor(
        model_name=file_name
    ),
    show_image=False
)
accuracy = results_accuracy(test_result)
print("Accuracy is {:.2f}%".format(accuracy * 100))
