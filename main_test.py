import time
import face_recognition

from preprocess.processor import get_processed_data
import numpy as np

from test_data import test_manager
from test_data.test_manager import results_accuracy

(new_X, new_X_raw, new_y,
 max_t_s_num,
 num_student,
 test_new_X, test_new_y) = get_processed_data()


# Averaging one person's all pictures to one
# If not using it, this is the same as using KNN with n=1 (I think)
# print('Use averaging')
# X_y_dict = training_set_to_dict(new_X, new_y)
# for k in X_y_dict:
#     X_y_dict[k] = [np.mean(X_y_dict[k], axis=0)]
# new_X, new_y = dict_to_training_set(X_y_dict)


def predict(face_encodings, tolerance=0.54, print_time=False):
    start = time.time()

    def predict_single(face_encoding):
        # See if the face is a match for the known face(s)
        face_dis = face_recognition.face_distance(new_X, face_encoding)

        name = 'Unknown'

        # matches = list(face_dis <= tolerance)
        #
        # # If a match was found in known_face_encodings, just use the first one.
        # if True in matches:
        #     first_match_index = matches.index(True)
        #     name = known_face_names[first_match_index]

        best_match_index = np.argmin(face_dis)
        if face_dis[best_match_index] <= tolerance:
            name = new_y[best_match_index]
        return name

    results = [predict_single(x) for x in face_encodings]
    if print_time:
        print('Made {} predictions in {:.3f}ms'.format(len(face_encodings), (time.time() - start) * 1000))

    return results


start = time.time()
for tolerance in np.arange(0.40, 0.61, 0.01):
    test_result = test_manager.test(
        predict_fn=
        lambda face_encodings: predict(face_encodings, tolerance=tolerance, print_time=False),
        show_image=False,
        print_info=False
    )
    accuracy = results_accuracy(test_result)
    print("At a tolerance of {:.2f}, accuracy is {:.2f}%".format(tolerance, accuracy * 100))
print("Test finished in {:.3f}ms".format((time.time() - start) * 1000))
