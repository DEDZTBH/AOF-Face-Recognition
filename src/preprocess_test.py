from src.preprocess import get_face_pics, training_set_to_dict, max_training_set_num, get_equal_number_training_set, \
    dict_to_training_set, get_encoding_for_known_face
from src.util import transform_2017_photos, save, load

recover = True
file_name = 'preprocess_test_0_100'
num_jitters = 100

if recover:
    (new_X, new_X_raw, new_y, max_t_s_num, num_student) = load(file_name)
else:
    known_faces, known_names = get_face_pics('2017 photos', file_name_transform=transform_2017_photos)
    X_y_dict = training_set_to_dict(known_faces, known_names)
    num_student = len(X_y_dict.keys())
    max_t_s_num = max_training_set_num(X_y_dict)
    new_X_y_dict = get_equal_number_training_set(X_y_dict, max_t_s_num)
    new_X_raw, new_y = dict_to_training_set(new_X_y_dict, shuffle_training_set=True)
    new_X = get_encoding_for_known_face(new_X_raw, num_jitters=num_jitters)
    save((new_X, new_X_raw, new_y, max_t_s_num, num_student), file_name)


