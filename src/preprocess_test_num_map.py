from src.preprocess import get_face_pics, training_set_to_dict, max_training_set_num, get_equal_number_training_set, \
    dict_to_training_set, get_encoding_for_known_face
from src.util import transform_2017_photos, save, load, dict_keys_map_to_numbers

recover = True
file_name = 'preprocess_test_num_map'

if recover:
    (new_X_num, num_map, new_y_num, max_t_s_num, num_student) = load(file_name)
else:
    known_faces, known_names = get_face_pics('2017 photos', file_name_transform=transform_2017_photos)
    X_y_dict = training_set_to_dict(known_faces, known_names)
    num_student = len(X_y_dict.keys())
    max_t_s_num = max_training_set_num(X_y_dict)
    new_X_y_dict = get_equal_number_training_set(X_y_dict, max_t_s_num)
    num_map, new_dict = dict_keys_map_to_numbers(new_X_y_dict)
    new_X_raw, new_y_num = dict_to_training_set(new_dict, max_t_s_num, shuffle_training_set=True)
    new_X_num = get_encoding_for_known_face(new_X_raw)
    save((new_X_num, num_map, new_y_num, max_t_s_num, num_student), file_name)
