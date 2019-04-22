from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.contrib.factorization import KMeans

from src.preprocess.processor_num_map import get_processed_data
from src.util.util import save

# Parameters
num_steps = 5000
# k = num_student
# num_classes = num_student
num_features = 128


def run_k_means(
        num_steps=num_steps,
        k=None,
        num_classes=None,
        num_features=num_features,
        keep_session=True
):
    (new_X_num, num_map, new_y_num,
     max_t_s_num,
     num_student,
     orig_new_X_num, orig_new_y) = get_processed_data()

    if k is None:
        k = round(num_student * 1.35914091423)
        print('Choosing {} clusters for {} students, {} samples'.format(k, num_student, len(new_y_num)))

    if num_classes is None:
        num_classes = num_student

    full_data_x = np.asarray(new_X_num)

    # Input
    X = tf.placeholder(tf.float32, shape=[None, num_features])
    # Labels (for assigning a label to a centroid and testing)
    y = tf.placeholder(tf.float32, shape=[None, num_classes])

    # K-Means Parameters
    kmeans = KMeans(inputs=X,
                    num_clusters=k,
                    distance_metric='squared_euclidean',
                    use_mini_batch=True)

    # Build KMeans graph
    training_graph = kmeans.training_graph()
    if len(training_graph) > 6:  # Tensorflow 1.4+
        (all_scores, cluster_idx, scores, cluster_centers_initialized,
         cluster_centers_var, init_op, train_op) = training_graph
    else:
        (all_scores, cluster_idx, scores, cluster_centers_initialized,
         init_op, train_op) = training_graph

    cluster_idx = cluster_idx[0]  # fix for cluster_idx being a tuple
    avg_distance = tf.reduce_mean(scores)

    saver = tf.train.Saver()

    # Start TensorFlow session
    sess = tf.Session()

    # Run the initializer
    sess.run(tf.global_variables_initializer(),
             feed_dict={X: full_data_x})
    sess.run(init_op, feed_dict={X: full_data_x})

    one_hot_y = sess.run(tf.one_hot(new_y_num, num_student))
    test_one_hot_y = sess.run(tf.one_hot(orig_new_y, num_student))

    # Training
    for i in range(1, num_steps + 1):
        _, d, idx = sess.run([train_op, avg_distance, cluster_idx],
                             feed_dict={X: full_data_x})
        if i % 500 == 0 or i == 1:
            print("Step %i, Avg Distance: %f" % (i, d))

    # Assign a label to each centroid
    # Count total number of labels per centroid, using the label of each training
    # sample to their closest centroid (given by 'idx')
    counts = np.zeros(shape=(k, num_classes))
    for i in range(len(idx)):
        counts[idx[i]] += one_hot_y[i]
    # Assign the most frequent label to the centroid
    # labels_map_np = [np.argmax(c) for c in counts]
    # Different strategy
    labels_map_np = [np.random.choice(np.argwhere(c == np.amax(c)).flatten()) for c in counts]

    labels_map = tf.convert_to_tensor(labels_map_np)

    # Evaluation ops
    # Lookup: centroid_id -> label
    cluster_label = tf.nn.embedding_lookup(labels_map, cluster_idx)
    # Compute accuracy
    correct_prediction = tf.equal(cluster_label, tf.cast(tf.argmax(y, 1), tf.int32))
    accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Test Model
    test_x, test_y = orig_new_X_num, test_one_hot_y
    print("Test Accuracy:", sess.run(accuracy_op, feed_dict={X: test_x, y: test_y}))

    save_path = saver.save(sess, "models/model.ckpt")
    print("Model saved in path: %s" % save_path)

    save(labels_map_np, 'labels_map_np')

    if not keep_session:
        sess.close()

    return k, num_classes, sess


if __name__ == '__main__':
    run_k_means(keep_session=False)
