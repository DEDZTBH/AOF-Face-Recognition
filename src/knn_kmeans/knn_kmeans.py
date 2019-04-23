from os import path

import tensorflow as tf

from src.knn_kmeans.kmeans import num_features, run_k_means
from src.knn import knn_generate, predict
from src.preprocess.processor_num_map import get_processed_data, get_file_name
from src.test_data import test_data
from src.util.util import load, decode_num_map, load_or_create

restore = False

(new_X_num, num_map, new_y_num,
 max_t_s_num,
 num_student,
 test_new_X_num, test_new_y) = get_processed_data()

n = round(max_t_s_num / 2)
print('Using n of {}'.format(n))

if restore:
    # Variable = tf.get_variable("Variable", shape=[k], dtype='int64')
    labels_num, k = load('labels_map_np,k', folder=path.join('pkl', 'kmeans'))
    clusters = tf.get_variable("clusters", shape=[k, num_features])
    saver = tf.train.Saver()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, "tf_models/model.ckpt")
    print("model restored")
else:
    k, num_classes, labels_num, sess = run_k_means(keep_session=True)
    clusters = [v for v in tf.trainable_variables() if v.name == 'clusters:0'][0]
clusters_np = clusters.eval(sess)
# Variable_np = Variable.eval(sess)
sess.close()

labels = decode_num_map(labels_num, num_map)

n = round(max_t_s_num / 2)
print('Using n of {}'.format(n))

extra = '1719_{}'.format(get_file_name())

knn_trained = load_or_create('knn_{}_{}'.format(extra, n),
                             create_fn=lambda: knn_generate(clusters_np, labels, n_neighbors=n, verbose=True),
                             folder=path.join('pkl', 'knn'))

test_data.test(
    predict_fn=
    lambda face_encodings: predict(face_encodings, knn_trained,
                                   distance_threshold=0.52,
                                   n_neighbors=n),
    show_image=False
)
