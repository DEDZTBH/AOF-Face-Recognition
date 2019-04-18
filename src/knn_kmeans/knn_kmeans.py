import face_recognition
import tensorflow as tf
from PIL import Image

from src.knn_kmeans.kmeans import k, num_features
from src.knn import knn_generate, predict, show_prediction_labels_on_image
from src.preprocess.processor_num_map import get_processed_data
from src.util.util import load, decode_num_map

(new_X_num, num_map, new_y_num,
 max_t_s_num,
 num_student,
 test_new_X_num, test_new_y) = get_processed_data()

# Variable = tf.get_variable("Variable", shape=[k], dtype='int64')
clusters = tf.get_variable("clusters", shape=[k, num_features])

saver = tf.train.Saver()
# with tf.Session() as sess:
sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver.restore(sess, "models/model.ckpt")
print("model restored")

clusters_np = clusters.eval(sess)
# Variable_np = Variable.eval(sess)

sess.close()

labels_num = load('labels_map_np')

labels = decode_num_map(labels_num, num_map)

knn_trained = knn_generate(clusters_np, labels, verbose=True)

# predict([known_face_encodings[0]], knn_trained)

unknown_image = face_recognition.load_image_file('data/unknown/51341390_10156668527250236_6458268350773460992_o.jpg')
# Find all the faces and face encodings in the unknown image
face_locations = face_recognition.face_locations(unknown_image)
face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

pil_image = Image.fromarray(unknown_image)

predictions = predict(face_encodings, knn_trained,
                      distance_threshold=0.54,
                      n_neighbors=3)

show_prediction_labels_on_image(pil_image, face_locations, predictions)
