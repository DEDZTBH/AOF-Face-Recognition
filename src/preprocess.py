import os
import face_recognition
import random
import dlib


def get_encodings(location="known", file_name_transform=lambda x: x, num_jitters=1):
    # Create arrays of known face encodings and their names
    known_face_encodings = []
    known_face_names = []

    directory = os.fsencode(location)
    for filename in os.listdir(directory):
        p = filename.decode("utf-8")
        known_face_names.append(file_name_transform(p))
        known_face_encodings.append(face_recognition.face_encodings(
            face_recognition.load_image_file(
                os.path.join(directory, filename).decode("utf-8")),
            num_jitters=num_jitters)[0])

    return known_face_encodings, known_face_names


def get_encodings_from_pics(pics, num_jitters=1):
    return list(map(lambda pic: face_recognition.face_encodings(pic, num_jitters=num_jitters)[0], pics))


def get_face_pics(location="known", file_name_transform=lambda x: x):
    # Create arrays of known face encodings and their names
    known_faces = []
    known_names = []

    directory = os.fsencode(location)
    for filename in os.listdir(directory):
        p = filename.decode("utf-8")
        known_names.append(file_name_transform(p))
        known_faces.append(face_recognition.load_image_file(
            os.path.join(directory, filename).decode("utf-8")))

    return known_faces, known_names


def training_set_to_dict(X, y):
    X_y_dict = {}
    if len(X) != len(y):
        raise ValueError("X and y are not same length")
    for i in len(X):
        if y[i] not in X_y_dict:
            X_y_dict[y[i]] = []
        X_y_dict[y[i]].append(X[i])


def dict_to_training_set(X_y_dict, max_exist_training_set_num=None, shuffle_keys=False):
    if max_exist_training_set_num is None:
        max_exist_training_set_num = max_training_set_num(X_y_dict)
    X = []
    y = []

    if shuffle_keys:
        the_keys = random.shuffle(X_y_dict.keys())
    else:
        the_keys = X_y_dict.keys()

    for key in the_keys:
        X += X_y_dict[key]
        for i in range(max_exist_training_set_num):
            y += the_keys

    return X, y


def max_training_set_num(X_y_dict):
    return max(list(map(lambda x: len(x), X_y_dict.values())))


def get_equal_number_training_set(X_y_dict, max_exist_training_set_num=None, generate_extra_for_each=0):
    if max_exist_training_set_num is None:
        max_exist_training_set_num = max_training_set_num(X_y_dict)
    for key in X_y_dict.keys():
        generate_list = []
        for i in range(max_exist_training_set_num - len(X_y_dict[key]) + generate_extra_for_each):
            generate_list.append(dlib.jitter_image(random.choice(X_y_dict[key])))
        X_y_dict[key] += generate_list
    return X_y_dict
