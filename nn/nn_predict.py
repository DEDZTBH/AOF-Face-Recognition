import pickle
import time
from os import path

import numpy as np

from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential, save_model, load_model
from tensorflow.python.keras.layers import Dense

from util.predictor import EncodingsPredictor, get_param, get_param_default

epochs = 1000
batch_size = 64

extra = '_tanh'

file_name = 'nn_y_{}_{}{}'.format(epochs, batch_size, extra)
file_path = path.join('data', 'model', 'nn', '{}.hdf5'.format(file_name))


def train_nn(new_X_num, new_y_num, num_student, num_map, save=False):
    start = time.time()

    num_classes = num_student

    recog_model = Sequential([
        Dense(128, activation="tanh", input_dim=128),
        Dense(num_classes, activation="softmax")
    ])

    recog_model.compile(loss=keras.losses.sparse_categorical_crossentropy,
                        optimizer='adam',
                        # metrics=['accuracy']
                        )

    training_history = recog_model.fit(
        np.array(new_X_num),
        np.array(new_y_num),
        batch_size=batch_size,
        epochs=epochs
    )

    # # summarize history for accuracy
    # plt.plot(training_history.history['acc'])
    # # plt.plot(training_history.history['val_acc'])
    # plt.title('model accuracy')
    # plt.ylabel('accuracy')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # plt.show()
    # # summarize history for loss
    # plt.plot(training_history.history['loss'])
    # # plt.plot(training_history.history['val_loss'])
    # plt.title('model loss')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # plt.show()

    print('Trained NN model in {:.3f}ms'.format((time.time() - start) * 1000))

    if save:
        save_model(
            model=recog_model,
            filepath=file_path
        )
        with open('{}.num_map.pkl'.format(file_path), 'wb') as file:
            pickle.dump(num_map, file)

    return recog_model, num_map


def predict(arr_face, recog_model, num_map, threshold, print_time=False):
    if print_time:
        start = time.time()
    processed_results = recog_model.predict(np.array(arr_face))
    valid_names = []
    for result in processed_results:
        valid_idx = np.argmax(result)
        if result[valid_idx] < threshold:
            valid_names.append('Unknown')
        else:
            valid_names.append(num_map[valid_idx])
    if print_time:
        print('Made {} predictions in {:.3f}ms'.format(len(arr_face), (time.time() - start) * 1000))
    return valid_names


def get_file_name():
    return file_name


def get_file_path():
    return file_path


class NNPredictor(EncodingsPredictor):
    def __init__(self, **kwargs):
        self.model_name = get_param('model_name', kwargs)
        file_path = path.join('data', 'model', 'nn', '{}.hdf5'.format(self.model_name))
        self.recog_model = load_model(file_path)
        with open('{}.num_map.pkl'.format(file_path), 'rb') as file:
            self.num_map = pickle.load(file)
        self.tolerance = get_param_default('tolerance', 0.54, kwargs)
        self.print_time = get_param_default('print_time', False, kwargs)

    def predict(self, face_encodings):
        return predict(arr_face=face_encodings,
                       recog_model=self.recog_model,
                       threshold=self.tolerance,
                       num_map=self.num_map,
                       print_time=self.print_time)
