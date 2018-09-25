import tensorflow as tf

print(tf.__version__)


def some_method(data):
    print(data.get_shape())
    a = data[:, 0:2]
    print(a.get_shape())
    c = data[:, 1:3]
    print(c.get_shape())
    s = (a + c)
    return tf.sqrt(tf.matmul(s, tf.transpose(s)))


with tf.Session() as sess:
    fake_data = tf.constant([5.0, 3.0, 7.1])
    print(sess.run(some_method(fake_data)))


def some_method(data):
    print(data.get_shape())
    a = data[:, 0:2]
    print(a.get_shape())
    c = data[:, 1:3]
    print(c.get_shape())
    s = (a + c)
    return tf.sqrt(tf.matmul(s, tf.transpose(s)))


with tf.Session() as sess:
    fake_data = tf.constant([5.0, 3.0, 7.1])
    fake_data = tf.expand_dims(fake_data, 0)
    print(sess.run(some_method(fake_data)))
