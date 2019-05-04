import re
import numpy as np

regex_exp = r'^[0-9.]*([A-Za-z \-\'\’]+\.[A-Za-z \-\'\’]+)\..*$'


def transform_yearbook_photos(filename):
    try:
        p = re.match(regex_exp, filename).group(1).replace('.', ' ')
    except AttributeError:
        raise AttributeError('{} in wrong format'.format(filename))
    return p


def dict_keys_map_to_numbers(dic, generate_new_dict=True, existing_keys_map=None):
    new_dict = {}

    if existing_keys_map is not None:
        for key in dic.keys():
            if generate_new_dict:
                new_dict[existing_keys_map.index(key)] = dic[key]
        return new_dict, existing_keys_map

    else:
        dict_num_map = []
        for key in dic.keys():
            if generate_new_dict:
                new_dict[len(dict_num_map)] = dic[key]
            dict_num_map.append(key)
        return new_dict, dict_num_map


def decode_num_map(encoded_arr, num_map):
    return [num_map[i] for i in encoded_arr]


def random_rows(A, num_rows):
    return A[np.random.choice(A.shape[0], num_rows, replace=False), :]
