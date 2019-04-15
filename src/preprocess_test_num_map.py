from src.preprocess import get_face_pics, training_set_to_dict, max_training_set_num, get_equal_number_training_set, \
    dict_to_training_set, get_encoding_for_known_face
from src.util import transform_2017_photos, save, load, dict_keys_map_to_numbers

# from PIL import Image

recover = True
file_name = 'preprocess_test_num_map_2_100'
generate_extra_for_each = 2
encoding_jitters = 100

if recover:
    (new_X_num, num_map, new_y_num,
     max_t_s_num,
     num_student,
     orig_new_X_num, orig_new_y) = load(file_name)
else:
    # known_faces_full, known_names = get_face_pics('2017 photos',
    #                                               file_name_transform=transform_2017_photos,
    #                                               face_only=False)
    known_faces, known_names = get_face_pics('2017 photos',
                                             file_name_transform=transform_2017_photos)

    # known_faces = known_faces[0:5]
    # known_names = known_names[0:5]

    # X_y_dict = training_set_to_dict(known_faces_full, known_names)
    X_y_dict = training_set_to_dict(known_faces, known_names)
    num_student = len(X_y_dict.keys())
    max_t_s_num = max_training_set_num(X_y_dict)
    new_X_y_dict = get_equal_number_training_set(X_y_dict, max_t_s_num, generate_extra_for_each=generate_extra_for_each,
                                                 copy_dict=True)
    max_t_s_num += generate_extra_for_each

    new_dict, num_map = dict_keys_map_to_numbers(new_X_y_dict)
    orig_new_dict, _ = dict_keys_map_to_numbers(X_y_dict, existing_keys_map=num_map)

    new_X_raw, new_y_num = dict_to_training_set(new_dict, shuffle_training_set=True)
    orig_new_X_raw, orig_new_y = dict_to_training_set(orig_new_dict, shuffle_training_set=False)

    # for i in new_X_raw:
    #     Image.fromarray(i).show()

    # new_X_num = get_encoding_for_known_face(new_X_raw, rescan=True, num_jitters=encoding_jitters)
    # orig_new_X_num = get_encoding_for_known_face(orig_new_X_raw, rescan=True, num_jitters=encoding_jitters)
    new_X_num = get_encoding_for_known_face(new_X_raw, rescan=False, num_jitters=encoding_jitters)
    orig_new_X_num = get_encoding_for_known_face(orig_new_X_raw, rescan=False, num_jitters=encoding_jitters)
    save((new_X_num, num_map, new_y_num,
          max_t_s_num,
          num_student,
          orig_new_X_num, orig_new_y)
         , file_name)
