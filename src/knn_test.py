from src.preprocess import get_encodings
from src.knn import knn_generate, predict, show_prediction_labels_on_image
import face_recognition
from PIL import Image
import math
from src.preprocess_test import new_X, new_y, max_t_s_num

from src.util import transform_2017_photos, load, save

# save_name = 'saved_augmented'
# recover = False
# if recover:
#     (known_face_encodings, known_face_names) = load(save_name)
# else:
#     known_face_encodings, known_face_names = get_encodings('2017 photos', file_name_transform=transform_2017_photos,
#                                                            num_jitters=0)
#     save((known_face_encodings, known_face_names), save_name)

knn_trained = knn_generate(new_X, new_y, verbose=True)

# predict([known_face_encodings[0]], knn_trained)

unknown_image = face_recognition.load_image_file('unknown/51341390_10156668527250236_6458268350773460992_o.jpg')
# Find all the faces and face encodings in the unknown image
face_locations = face_recognition.face_locations(unknown_image)
face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

pil_image = Image.fromarray(unknown_image)

predictions = predict(face_encodings, knn_trained,
                      distance_threshold=0.52,
                      n_neighbors=max_t_s_num * 2 - 1)

show_prediction_labels_on_image(pil_image, face_locations, predictions)
