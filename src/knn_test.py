from src.knn import knn_generate, predict, show_prediction_labels_on_image
import face_recognition
from PIL import Image
from src.preprocess.processor import get_processed_data

(new_X, new_X_raw, new_y,
 max_t_s_num,
 num_student,
 test_new_X, test_new_y) = get_processed_data()

n = round(max_t_s_num / 2)
print('Using n of {}'.format(n))

knn_trained = knn_generate(new_X, new_y, n_neighbors=n, verbose=True)

unknown_image = face_recognition.load_image_file('data/unknown/51341390_10156668527250236_6458268350773460992_o.jpg')
# Find all the faces and face encodings in the unknown image
face_locations = face_recognition.face_locations(unknown_image)
face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

pil_image = Image.fromarray(unknown_image)

predictions = predict(face_encodings, knn_trained,
                      distance_threshold=0.52,
                      n_neighbors=n)

show_prediction_labels_on_image(pil_image, face_locations, predictions)
