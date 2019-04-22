from src.preprocess.preprocess import get_face_pics, training_set_to_dict, max_training_set_num, \
    get_equal_number_training_set, \
    dict_to_training_set, get_encoding_for_known_face, shuffle_training_data
from src.util.util import transform_2017_photos, save, load
import copy

recover = None
generate_extra_for_each = 0
encoding_jitters = 100
neq = False
orig_jitters = 1


def get_processed_data(recover=recover, generate_extra_for_each=generate_extra_for_each,
                       encoding_jitters=encoding_jitters, orig_jitters=orig_jitters, neq=neq):
    file_name = 'preprocess_test_{}_{}{}'.format(generate_extra_for_each, encoding_jitters, '_neq' if neq else '')
    magic_obj = None

    if recover is None:
        try:
            magic_obj = load(file_name)
            recover = True
        except FileNotFoundError:
            recover = False

    if recover:
        if magic_obj is None:
            magic_obj = load(file_name)
    else:
        known_faces, known_names = get_face_pics('data/known', file_name_transform=transform_2017_photos)
        X_y_dict = training_set_to_dict(known_faces, known_names)
        num_student = len(X_y_dict.keys())
        max_t_s_num = max_training_set_num(X_y_dict)

        if not neq:
            new_X_y_dict = get_equal_number_training_set(X_y_dict, max_t_s_num,
                                                         generate_extra_for_each=generate_extra_for_each,
                                                         copy_dict=True)
            max_t_s_num += generate_extra_for_each
        else:
            new_X_y_dict = X_y_dict

        new_X_raw, new_y = dict_to_training_set(new_X_y_dict, shuffle_training_set=False)
        if not neq:
            new_X_raw, new_y = shuffle_training_data(new_X_raw, new_y)
            orig_new_X_raw, orig_new_y = dict_to_training_set(X_y_dict, shuffle_training_set=False)
        else:
            orig_new_X_raw = copy.deepcopy(new_X_raw)
            orig_new_y = copy.deepcopy(new_y)

        new_X = get_encoding_for_known_face(new_X_raw, rescan=False, num_jitters=encoding_jitters)
        orig_new_X = get_encoding_for_known_face(orig_new_X_raw, rescan=False, num_jitters=orig_jitters)

        magic_obj = (new_X, new_X_raw, new_y,
                     max_t_s_num,
                     num_student,
                     orig_new_X, orig_new_y)
        save(magic_obj, file_name)
    return magic_obj


if __name__ == '__main__':
    (new_X, new_X_raw, new_y,
     max_t_s_num,
     num_student,
     orig_new_X, orig_new_y) = get_processed_data()
