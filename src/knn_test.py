import pickle
from src.preprocess import get_encodings
import re
from src.knn import knn_generate, predict, show_prediction_labels_on_image
import face_recognition
from PIL import Image
import math


regex_exp = r'^[0-9.]*([A-Za-z \-\']+\.[A-Za-z \-\']+)\..*$'


def transform_2017_photos(filename):
    print(filename)
    p = re.match(regex_exp, filename).group(1).replace('.', ' ')
    print(p)
    return p


save_name = 'saved.pkl'
recover = True
if recover:
    with open(save_name, 'rb') as file:
        saved = pickle.load(file)
        (known_face_encodings, known_face_names) = saved
else:
    known_face_encodings, known_face_names = get_encodings('2017 photos', file_name_transform=transform_2017_photos)
    with open(save_name, 'wb') as file:
        pickle.dump((known_face_encodings, known_face_names), file)

knn_trained = knn_generate(known_face_encodings, known_face_names, verbose=True)

# predict([known_face_encodings[0]], knn_trained)

unknown_image = face_recognition.load_image_file('unknown/51341390_10156668527250236_6458268350773460992_o.jpg')
# Find all the faces and face encodings in the unknown image
face_locations = face_recognition.face_locations(unknown_image)
face_encodings = face_recognition.face_encodings(unknown_image, face_locations)
pil_image = Image.fromarray(unknown_image)

predictions = predict(face_encodings, knn_trained,
                      distance_threshold=0.54,
                      n_neighbors=int(round(math.sqrt(len(known_face_encodings)))))

show_prediction_labels_on_image(pil_image, face_locations, predictions)
