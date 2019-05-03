from knn.knn_predict import knn_generate, predict
from util.file import load
from util.general import decode_num_map
from os import path

import tensorflow as tf

from knn_kmeans.kmeans import num_features, run_k_means
from util.predictor import EncodingsPredictor, get_param, get_param_default


def knn_kmeans_generate(restore, n, num_map):
    if restore:
        # Variable = tf.get_variable("Variable", shape=[k], dtype='int64')
        labels_num, k = load('labels_map_np,k', folder=path.join('data', 'cache', 'kmeans'))
        clusters = tf.get_variable("clusters", shape=[k, num_features])
        saver = tf.train.Saver()
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, path.join('data', 'cache', 'kmeans', 'tf_model', 'model.ckpt'))
        print("model restored")
    else:
        k, num_classes, labels_num, sess = run_k_means(keep_session=True)
        clusters = [v for v in tf.trainable_variables() if v.name == 'clusters:0'][0]
    clusters_np = clusters.eval(sess)
    # Variable_np = Variable.eval(sess)
    sess.close()

    labels = decode_num_map(labels_num, num_map)

    return knn_generate(clusters_np, labels, n_neighbors=n, verbose=True)


class KNNKmeansPredictor(EncodingsPredictor):
    def __init__(self, **kwargs):
        self.knn_trained = load(
            filename=get_param('model_name', kwargs),
            folder=path.join('data', 'model', 'knn'))
        self.n = get_param('n', kwargs)
        self.tolerance = get_param_default('tolerance', 0.54, kwargs)
        self.print_time = get_param_default('print_time', False, kwargs)

    def predict(self, face_encodings):
        return predict(X_encodings=face_encodings,
                       knn_clf=self.knn_trained,
                       distance_threshold=self.tolerance,
                       n_neighbors=self.n,
                       print_time=self.print_time)
