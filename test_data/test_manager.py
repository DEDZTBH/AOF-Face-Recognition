import face_recognition
from PIL import Image, ImageFont, ImageDraw

from util.file import load_or_create, save
from os import path
import numpy as np

from util.general import random_rows
from util.predictor import EncodingsPredictor

font = ImageFont.truetype(path.join('fonts', 'Arial Bold.ttf'), 12)

test_data_file_name = 'test_data_640'
test_data_folder_name = path.join('data', 'test_data')

test_data_cache = None


def get():
    global test_data_cache
    if test_data_cache is None:
        test_data_cache = load_or_create(filename=test_data_file_name,
                                         create_fn=lambda: [],
                                         folder=test_data_folder_name)
    return test_data_cache


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
            false_positive = 0
            false_negative = 0
            for i in range(num_face):
                if predictions[i] == face_answers[i]:
                    correct_count += 1
                elif predictions[i] == 'Unknown':
                    false_negative += 1
                else:
                    false_positive += 1
            record.append({
                'file_name': file_name,
                'correct': correct_count,
                'false_positive': false_positive,
                'false_negative': false_negative,
                'total': num_face
            })
            if print_info:
                print('Recognition accuracy on {} is {:.2f}%'.format(file_name, (correct_count / num_face * 100)))
    return record


def test_predictor(predictor, show_image=False, print_info=True):
    if isinstance(predictor, EncodingsPredictor):
        return test(predict_fn=predictor.predict, show_image=show_image, print_info=print_info)
    else:
        raise ValueError('Need a EncodingsPredictor')


def results_accuracy(results):
    correct = 0
    total = 0
    for result in results:
        correct += result['correct']
        total += result['total']
    return correct / total


def results_stat(results):
    correct = 0
    false_positive = 0
    false_negative = 0
    total = 0
    for result in results:
        correct += result['correct']
        false_positive += result['false_positive']
        false_negative += result['false_negative']
        total += result['total']
    return {
        'accuracy': correct / total,
        'false_positive_rate': false_positive / total,
        'false_negative_rate': false_negative / total,
        'total': total
    }


def show_prediction_labels_on_image(X_img, face_locations, predictions):
    """
    Shows the face recognition results visually.
    :param X_img: pil image to be recognized
    :param predictions: results of the predict function
    :return:
    """
    draw = ImageDraw.Draw(X_img)

    # Loop through each face found in the unknown image
    for (top, right, bottom, left), name in zip(face_locations, predictions):
        # Draw a box around the face using the Pillow module
        draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

        draw.text((left + 6, bottom + np.random.randint(0, 20)), str(name),
                  fill=(0, 210, 0, 255), font=font)

    # Remove the drawing library from memory as per the Pillow docs
    del draw

    # Display the resulting image
    X_img.show()


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


def init_resize(analyze_file_name=analyze_file_name, max_pixel=1280, number_of_times_to_upsample=1, show_img=False):
    unknown_image = face_recognition.load_image_file(path.join('training_data', 'unknown', analyze_file_name))
    u_img = Image.fromarray(unknown_image)
    if u_img.width > u_img.height:
        u_img = u_img.resize((max_pixel, int(round(max_pixel * u_img.height / u_img.width))), resample=Image.BILINEAR)
    else:
        u_img = u_img.resize((int(round(max_pixel * u_img.width / u_img.height)), max_pixel), resample=Image.BILINEAR)
    if show_img:
        u_img.show()
    unknown_image = np.array(u_img)
    face_locations = face_recognition.face_locations(unknown_image,
                                                     number_of_times_to_upsample=number_of_times_to_upsample)
    face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

    return unknown_image, face_locations, face_encodings


def test_these(pics_l_e, predictor, show_img=True):
    names = []
    for unknown_image, face_locations, face_encodings in pics_l_e:
        predictions = predictor.predict(face_encodings)
        names.append(predictions)
        if show_img:
            show_prediction_labels_on_image(Image.fromarray(unknown_image), face_locations, predictions)
    return names


def add_test_these(pics_l_e, names):
    for (_, face_locations, face_encodings), name in zip(pics_l_e, names):
        add(test_data_file_name, face_locations, face_encodings, name)


def randomly_pick_encodings(num_range=None):
    if num_range is None:
        num_range = [1]
    test_data = get()
    all_encodings = []
    for test_datum in test_data:
        all_encodings += test_datum[2]
    return [random_rows(np.array(all_encodings), a, replace=True) for a in num_range]


def generate_validation_data(exclude_unknown=True, num_map=None):
    result = ([], [])
    num_map = np.array(num_map)
    for picture in get():
        for encodings, name in zip(picture[2], picture[3]):
            if not exclude_unknown or name != 'Unknown':
                result[0].append(encodings)
                if num_map is not None:
                    result[1].append(np.where(num_map == name)[0][0])
                else:
                    result[1].append(name)
    return np.array(result[0]), np.array(result[1])
