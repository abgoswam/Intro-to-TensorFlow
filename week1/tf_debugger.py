import tensorflow as tf
from tensorflow.python import debug as tf_debug


def some_method(a, b):
    b = tf.cast(b, tf.float32)
    s = (a / b)
    s2 = tf.matmul(s, tf.transpose(s))
    return tf.sqrt(s2)


with tf.Session() as sess:
    fake_a = [
        [5.0, 3.0, 7.1],
        [2.3, 4.1, 4.8],
    ]
    fake_b = [
        [2, 0, 5],
        [2, 8, 7]
    ]

    a = tf.placeholder(tf.float32, shape=[2, 3])
    b = tf.placeholder(tf.int32, shape=[2, 3])
    k = some_method(a, b)

    # sess = tf_debug.LocalCLIDebugWrapperSession(sess, ui_type="readline")
    sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
    print(sess.run(k, feed_dict={a: fake_a, b: fake_b}))
