import tensorflow as tf

i = tf.constant(0)
loop = tf.while_loop(lambda i: i <= 0, lambda i: i + 1, [i])


with tf.Session():
    # 10
    print(loop.eval())
