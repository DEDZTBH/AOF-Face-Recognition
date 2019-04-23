from src.knn import knn_generate, predict
from src.preprocess.processor import get_processed_data, get_file_name
from os import path

from src.test_data import test_data
from src.util.util import load_or_create

(new_X, new_X_raw, new_y,
 max_t_s_num,
 num_student,
 test_new_X, test_new_y) = get_processed_data()

n = round(max_t_s_num / 2)
print('Using n of {}'.format(n))

extra = '1719_{}'.format(get_file_name())

knn_trained = load_or_create('knn_{}_{}'.format(extra, n),
                             create_fn=lambda: knn_generate(new_X, new_y, n_neighbors=n, verbose=True),
                             folder=path.join('pkl', 'knn'))

test_data.test(
    predict_fn=
    lambda face_encodings: predict(face_encodings, knn_trained,
                                   distance_threshold=0.52,
                                   n_neighbors=n),
    show_image=True
)
