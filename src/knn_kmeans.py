import tensorflow as tf

from src.kmeans import k, num_features

Variable = tf.get_variable("Variable", shape=[k], dtype='int64')
clusters = tf.get_variable("clusters", shape=[k, num_features])
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, "models/model.ckpt")
    print("model restored")
