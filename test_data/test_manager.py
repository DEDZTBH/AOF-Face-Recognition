import face_recognition
from PIL import Image

from knn import show_prediction_labels_on_image
from util.general import load_or_create, save
from os import path

test_data_file_name = 'test_data'
test_data_folder_name = path.join('data', 'test_data')


def get():
    return load_or_create(filename=test_data_file_name,
                          create_fn=lambda: [],
                          folder=test_data_folder_name)


def add(file_name, face_locations, face_encodings, face_answers):
    test_data = get()
    test_data.append((file_name, face_locations, face_encodings, face_answers))
    save(test_data, filename=test_data_file_name, folder=test_data_folder_name)


def delete(file_name):
    test_data = get()
    test_data = [data for data in test_data if data[0] != file_name]
    save(test_data, filename=test_data_file_name, folder=test_data_folder_name)


def test(predict_fn, show_image=False, print_info=True):
    record = []
    for (file_name, face_locations, face_encodings, face_answers) in get():

        predictions = predict_fn(face_encodings)

        if show_image:
            unknown_image = face_recognition.load_image_file(path.join('training_data', 'unknown', file_name))
            pil_image = Image.fromarray(unknown_image)
            show_prediction_labels_on_image(pil_image, face_locations, predictions)

        num_face = len(face_answers)
        if num_face == len(predictions):
            correct_count = 0
            for i in range(num_face):
                if predictions[i] == face_answers[i]:
                    correct_count += 1
            record.append((file_name, correct_count, num_face))
            if print_info:
                print('Recognition accuracy on {} is {:.2f}%'.format(file_name, (correct_count / num_face * 100)))
    return record


def results_accuracy(results):
    correct = 0
    total = 0
    for result in results:
        correct += result[1]
        total += result[2]
    return correct / total


analyze_file_name = '3.jpg'


def init(analyze_file_name=analyze_file_name, number_of_times_to_upsample=1):
    unknown_image = face_recognition.load_image_file(path.join('training_data', 'unknown', analyze_file_name))

    # Find all the faces and face encodings in the unknown image
    face_locations = face_recognition.face_locations(unknown_image,
                                                     number_of_times_to_upsample=number_of_times_to_upsample)
    face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

    return unknown_image, face_locations, face_encodings


if __name__ == '__main__':
    unknown_image, face_locations, face_encodings = init()
