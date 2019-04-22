import os
import face_recognition
import tensorflow as tf
import numpy as np
from PIL import Image
import copy


def get_encodings(location="data/known", file_name_transform=lambda x: x, num_jitters=1):
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
    return [face_recognition.face_encodings(pic, num_jitters=num_jitters)[0] for pic in pics]


def get_face_locations(img_arr):
    # print(img_arr.shape)
    # return face_recognition.face_locations(img_arr, model="cnn")
    return face_recognition.face_locations(img_arr)


def get_face(img):
    [top, right, bottom, left] = get_face_locations(np.array(img))[0]
    return img.crop((left, top, right, bottom))


def get_faces(imgs):
    return [get_face(img) for img in imgs]


def get_face_pics(location="data/known", file_name_transform=lambda x: x, face_only=True, resize_to_square=None):
    if resize_to_square is None:
        resize_to_square = face_only

    # Create arrays of known face encodings and their names
    known_faces = []
    known_names = []

    directory = os.fsencode(location)

    global i
    for idx, filename in enumerate(os.listdir(directory)):
        p = filename.decode("utf-8")
        name = file_name_transform(p)
        known_names.append(name)

        raw_photo_to_add = Image.open(os.path.join(directory, filename).decode("utf-8")).convert('RGB')

        if face_only:
            raw_photo_to_add = get_face(raw_photo_to_add)

        if resize_to_square:
            (width, height) = raw_photo_to_add.size
            if width != height:
                new_width_height = int(np.average((width, height)))
                raw_photo_to_add = raw_photo_to_add.resize((new_width_height, new_width_height))
                # print('Image of {} has size {}x{}, resize to {}'.format(name, width, height, new_width_height))

        known_faces.append(np.array(raw_photo_to_add))
        i = idx + 1
        if i % 10 == 0:
            print('Processed {} pictures'.format(i))

    print('Processed {} pictures'.format(i))
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


def shuffle_training_data(X, y):
    matrix = np.asmatrix([X, y]).transpose()
    np.random.shuffle(matrix)
    matrix = matrix.transpose()
    X = matrix[0].A1
    y = matrix[1].A1
    return X, y


def dict_to_training_set(X_y_dict, shuffle_training_set=False):
    X = []
    y = []

    for key in X_y_dict.keys():
        X += X_y_dict[key]
        for _ in X_y_dict[key]:
            y.append(key)

    if shuffle_training_set:
        X, y = shuffle_training_data(X, y)

    return X, y


def max_training_set_num(X_y_dict):
    return max([len(x) for x in X_y_dict.values()])


jitter_generator = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=20,
                                                                   channel_shift_range=32)


def generate_gitter_image(rand_photo):
    return jitter_generator.random_transform(rand_photo).astype(np.uint8)


def get_equal_number_training_set(X_y_dict, max_exist_training_set_num=None, generate_extra_for_each=0,
                                  copy_dict=False):
    if max_exist_training_set_num is None:
        max_exist_training_set_num = max_training_set_num(X_y_dict)

    if copy_dict:
        dic = copy.deepcopy(X_y_dict)
    else:
        dic = X_y_dict

    total_num = len(dic.keys())
    for idx, key in enumerate(dic.keys()):
        generate_list = []
        photos = dic[key]
        need_to_add_num = max_exist_training_set_num - len(photos) + generate_extra_for_each
        for _ in range(need_to_add_num):
            rand_photo = photos[np.random.randint(len(photos))]
            generate_list.append(generate_gitter_image(rand_photo))
        dic[key] += generate_list

        if (idx + 1) % 10 == 0 or idx + 1 == total_num:
            print('Group processed {}/{}'.format(idx + 1, total_num))
    return dic


def get_encoding_for_known_face(imgs_array, rescan=False, num_jitters=1):
    total_num = len(imgs_array)

    def process(x, i):
        x_np = np.array(x)
        if rescan:
            face_locs = get_face_locations(x_np)
        else:
            face_locs = [(0, x_np.shape[1] - 1, x_np.shape[0] - 1, 0)]
        found_face = face_recognition.face_encodings(x_np, face_locs, num_jitters=num_jitters)
        if len(found_face) != 1:
            Image.fromarray(x_np).show()
            raise ValueError("Found {} face(s) in picture".format(len(found_face)))

        if i % 10 == 0 or i == total_num:
            print('Encoding generated {}/{}'.format(i, total_num))

        return found_face[0]

    return [process(img, i + 1) for i, img in enumerate(imgs_array)]


def resize_imgs(imgs, height=140, width=140, asarray=True):
    if asarray:
        processed_imgs = [Image.fromarray(x) for x in imgs]

        def process(img):
            return np.asarray(img.resize((height, width)))
    else:
        processed_imgs = imgs

        def process(img):
            return img.resize((height, width))

    return [process(img) for img in processed_imgs]
