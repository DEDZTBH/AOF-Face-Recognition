from src.preprocess.preprocess import training_set_to_dict, dict_to_training_set, shuffle_training_data
from src.util.util import save, load, dict_keys_map_to_numbers
import src.preprocess.processor as raw_processor
import copy

recover = None
generate_extra_for_each = 0
encoding_jitters = 100
neq = False
orig_jitters = 1


def get_processed_data(recover=recover, generate_extra_for_each=generate_extra_for_each,
                       encoding_jitters=encoding_jitters, orig_jitters=orig_jitters, neq=neq):
    file_name = 'preprocess_test_num_map_{}_{}{}'.format(generate_extra_for_each, encoding_jitters,
                                                         '_neq' if neq else '')
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
        (new_X, new_X_raw, new_y,
         max_t_s_num,
         num_student,
         orig_new_X, orig_new_y) = raw_processor.get_processed_data(
            recover=None,
            generate_extra_for_each=generate_extra_for_each,
            encoding_jitters=encoding_jitters,
            neq=neq,
            orig_jitters=orig_jitters
        )

        new_X_y_dict = training_set_to_dict(new_X, new_y)
        X_y_dict = training_set_to_dict(orig_new_X, orig_new_y)

        new_dict, num_map = dict_keys_map_to_numbers(new_X_y_dict)
        orig_new_dict, _ = dict_keys_map_to_numbers(X_y_dict, existing_keys_map=num_map)

        new_X_num, new_y_num = dict_to_training_set(new_dict, shuffle_training_set=False)
        if not neq:
            # Already shuffled, not need to shuffle again
            orig_new_X_num, orig_new_y = dict_to_training_set(orig_new_dict, shuffle_training_set=False)
        else:
            orig_new_X_num = copy.deepcopy(new_X_num)
            orig_new_y = copy.deepcopy(new_y_num)

        # Get Examples
        # for i in new_X_raw:
        #     Image.fromarray(i).show()

        magic_obj = (new_X_num, num_map, new_y_num,
                     max_t_s_num,
                     num_student,
                     orig_new_X_num, orig_new_y)

        save(magic_obj, file_name)

    return magic_obj


if __name__ == '__main__':
    (new_X_num, num_map, new_y_num,
     max_t_s_num,
     num_student,
     orig_new_X_num, orig_new_y) = get_processed_data()
