import time
from os import path

import numpy as np

from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential, save_model, load_model
from tensorflow.python.keras.layers import Dense

from preprocess.processor_num_map import get_processed_data
import matplotlib.pyplot as plt

from test_data import test_manager
from test_data.test_manager import results_accuracy
from util.general import load_or_create

(new_X_num, num_map, new_y_num,
 max_t_s_num,
 num_student,
 orig_new_X_num, orig_new_y) = get_processed_data()

epochs = 1000
batch_size = 64

file_name = 'nn_{}_{}'.format(epochs, batch_size)
file_path = path.join('data', 'model', 'nn', '{}.hdf5'.format(file_name))


def train(save=False):
    start = time.time()

    num_classes = num_student

    recog_model = Sequential([
        Dense(128, activation="relu", input_dim=128),
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

    if (save):
        save_model(
            model=recog_model,
            filepath=file_path
        )

    return recog_model


def predict(arr_face, recog_model, threshold, print_time=False):
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


if __name__ == '__main__':
    # recog_model = train(True)
    recog_model = load_model(
        file_path
    )

    test_result = test_manager.test(
        predict_fn=lambda arr_face:
        predict(arr_face, recog_model, 0.52, True),
        show_image=True
    )
    accuracy = results_accuracy(test_result)
    print("Accuracy is {:.2f}%".format(accuracy * 100))

    # start = time.time()
    # for tolerance in np.arange(0.40, 0.61, 0.01):
    #     test_result = test_manager.test(
    #         predict_fn=
    #         lambda face_encodings: predict(face_encodings, recog_model, tolerance, False),
    #         show_image=False,
    #         print_info=False
    #     )
    #     accuracy = results_accuracy(test_result)
    #     print("At a tolerance of {:.2f}, accuracy is {:.2f}%".format(tolerance, accuracy * 100))
    # print("Test finished in {:.3f}ms".format((time.time() - start) * 1000))
