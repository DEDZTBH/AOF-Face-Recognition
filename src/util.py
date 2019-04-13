import math
import re
import pickle


def face_distance_to_conf(face_distance, face_match_threshold):
    if face_distance > face_match_threshold:
        range = (1.0 - face_match_threshold)
        linear_val = (1.0 - face_distance) / (range * 2.0)
        return linear_val
    else:
        range = face_match_threshold
        linear_val = 1.0 - (face_distance / (range * 2.0))
        return linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))


regex_exp = r'^[0-9.]*([A-Za-z \-\']+\.[A-Za-z \-\']+)\..*$'


def transform_2017_photos(filename):
    # print(filename)
    p = re.match(regex_exp, filename).group(1).replace('.', ' ')
    # print(p)
    return p


def save(stuff, filename, ext='pkl'):
    with open('{}.{}'.format(filename, ext), 'wb') as file:
        pickle.dump(stuff, file)


def load(filename, ext='pkl'):
    with open('{}.{}'.format(filename, ext), 'rb') as file:
        return pickle.load(file)


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
