import tensorflow as tf

a = tf.constant([5, 3, 8])
b = tf.constant([3, -1, 2])

c = tf.add(a, b)
print(c)

with tf.Session() as sess:
    result = sess.run(c)
    print(result)

