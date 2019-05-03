from preprocess.preprocess_util import get_face_pics, training_set_to_dict, max_training_set_num, \
    get_equal_number_training_set, \
    dict_to_training_set, get_encoding_for_known_face, shuffle_training_data, generate_gitter_image
from util.general import transform_yearbook_photos
from util.file import load_or_create
import copy
from os import path

neq = True
aug = False
generate_extra_for_each = 0 \
    if not neq else 0
encoding_jitters = 100
orig_jitters = 1
file_name = None
extra = '_y'


def get_processed_data(generate_extra_for_each=generate_extra_for_each,
                       encoding_jitters=encoding_jitters, orig_jitters=orig_jitters, neq=neq, aug=aug):
    def _get_processed_data():
        known_faces, known_names = get_face_pics(path.join('training_data', 'known'),
                                                 file_name_transform=transform_yearbook_photos)
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

        if aug:
            new_X_raw = [generate_gitter_image(x) for x in new_X_raw]

        if not neq:
            new_X_raw, new_y = shuffle_training_data(new_X_raw, new_y)
            orig_new_X_raw, orig_new_y = dict_to_training_set(X_y_dict, shuffle_training_set=False)
        else:
            orig_new_X_raw = copy.deepcopy(new_X_raw)
            orig_new_y = copy.deepcopy(new_y)

        new_X = get_encoding_for_known_face(new_X_raw, rescan=False, num_jitters=encoding_jitters)
        orig_new_X = get_encoding_for_known_face(orig_new_X_raw, rescan=False, num_jitters=orig_jitters)

        return (new_X, new_y,
                max_t_s_num,
                num_student,
                orig_new_X, orig_new_y)

    global file_name
    file_name = 'preprocess{}_{}_{}{}{}'.format(extra, generate_extra_for_each, encoding_jitters, '_neq' if neq else '',
                                                '_aug' if aug else '')

    magic_obj = load_or_create(file_name, create_fn=_get_processed_data,
                               folder=path.join('data', 'cache', 'preprocess'))

    return magic_obj


def get_file_name():
    return file_name


if __name__ == '__main__':
    (new_X, new_y,
     max_t_s_num,
     num_student,
     orig_new_X, orig_new_y) = get_processed_data()
