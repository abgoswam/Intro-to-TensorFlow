import tensorflow as tf

print(tf.__version__)


def some_method(a, b):
    s = (a + b)
    return tf.sqrt(tf.matmul(s, tf.transpose(s)))


with tf.Session() as sess:
    fake_a = tf.constant([
        [5.0, 3.0, 7.1],
        [2.3, 4.1, 4.8],
    ])
    fake_b = tf.constant([
        [2, 4, 5],
        [2, 8, 7]
    ])
    print(sess.run(some_method(fake_a, fake_b)))


def some_method(a, b):
    b = tf.cast(b, tf.float32)
    s = (a + b)
    return tf.sqrt(tf.matmul(s, tf.transpose(s)))


with tf.Session() as sess:
    fake_a = tf.constant([
        [5.0, 3.0, 7.1],
        [2.3, 4.1, 4.8],
    ])
    fake_b = tf.constant([
        [2, 4, 5],
        [2, 8, 7]
    ])
    print(sess.run(some_method(fake_a, fake_b)))
