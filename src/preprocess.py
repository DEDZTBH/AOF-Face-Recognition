import os
import face_recognition
import dlib
import numpy as np
from PIL import Image


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


def get_face_pics(location="known", file_name_transform=lambda x: x, face_only=True, resize_to_square=None):
    if resize_to_square is None:
        resize_to_square = face_only

    # Create arrays of known face encodings and their names
    known_faces = []
    known_names = []

    directory = os.fsencode(location)
    for filename in os.listdir(directory):
        p = filename.decode("utf-8")
        name = file_name_transform(p)
        known_names.append(name)

        raw_photo_to_add = Image.open(os.path.join(directory, filename).decode("utf-8")).convert('RGB')

        if face_only:
            [top, right, bottom, left] = face_recognition.face_locations(np.array(raw_photo_to_add), model="cnn")[0]
            raw_photo_to_add = raw_photo_to_add.crop((left, top, right, bottom))

        if resize_to_square:
            (width, height) = raw_photo_to_add.size
            if width != height:
                new_width_height = int(np.average((width, height)))
                raw_photo_to_add = raw_photo_to_add.resize((new_width_height, new_width_height))
                print('Image of {} has size {}x{}, resize to {}'.format(name, width, height, new_width_height))

        known_faces.append(np.array(raw_photo_to_add))

    return known_faces, known_names


def training_set_to_dict(X, y):
    X_y_dict = {}
    if len(X) != len(y):
        raise ValueError("X and y are not same length")
    for i in range(len(X)):
        if y[i] not in X_y_dict:
            X_y_dict[y[i]] = []
        X_y_dict[y[i]].append(X[i])
    return X_y_dict


def dict_to_training_set(X_y_dict, max_exist_training_set_num=None, shuffle_training_set=False):
    if max_exist_training_set_num is None:
        max_exist_training_set_num = max_training_set_num(X_y_dict)
    X = []
    y = []

    for key in X_y_dict.keys():
        X += X_y_dict[key]
        for i in range(max_exist_training_set_num):
            y.append(key)

    if shuffle_training_set:
        matrix = np.asmatrix([X, y]).transpose()
        np.random.shuffle(matrix)
        matrix = matrix.transpose()
        X = matrix[0].A1
        y = matrix[1].A1

    return X, y


def max_training_set_num(X_y_dict):
    return max(list(map(lambda x: len(x), X_y_dict.values())))


def get_equal_number_training_set(X_y_dict, max_exist_training_set_num=None, generate_extra_for_each=0):
    if max_exist_training_set_num is None:
        max_exist_training_set_num = max_training_set_num(X_y_dict)
    for key in X_y_dict.keys():
        generate_list = []
        photos = X_y_dict[key]
        need_to_add_num = max_exist_training_set_num - len(photos) + generate_extra_for_each
        for i in range(need_to_add_num):
            rand_photo = photos[np.random.randint(len(photos))]
            # print(rand_photo.shape)
            generate_list += dlib.jitter_image(rand_photo)
        X_y_dict[key] += generate_list
    return X_y_dict


def get_encoding_for_known_face(imgs_array, rescan=False, num_jitters=0):
    def process(x):
        x_np = np.asarray(x)
        # print(x_np.shape)
        if rescan:
            face_locs = None
        else:
            face_locs = [[0, x_np.shape[1] - 1, x_np.shape[0] - 1, 1]]
        return face_recognition.face_encodings(x_np, face_locs, num_jitters=num_jitters)[0]

    return list(map(process, imgs_array))
