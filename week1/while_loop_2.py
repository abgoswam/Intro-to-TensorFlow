import tensorflow as tf


def f_x(a0, a1, a2, a3, a4, x):
    return a0 + (a1 * tf.pow(x, 1)) + (a2 * tf.pow(x, 2)) + (a3 * tf.pow(x, 3)) + (a4 * tf.pow(x, 4))


def f1_x(a1, a2, a3, a4, x):
    return a1 + (2 * a2 * tf.pow(x, 1)) + (3 * a3 * tf.pow(x, 2)) + (4 * a4 * tf.pow(x, 3))


def f2_x(a2, a3, a4, x):
    return (2 * a2) + (3 * 2 * a3 * tf.pow(x, 1)) + (4 * 3 * a4 * tf.pow(x, 2))


def step(a, x):
    a0, a1, a2, a3, a4 = a[0], a[1], a[2], a[3], a[4]

    f = f_x(a0, a1, a2, a3, a4, x)
    f1 = f1_x(a1, a2, a3, a4, x)
    f2 = f2_x(a2, a3, a4, x)

    _n = 2*f*f1
    _d = 2*tf.pow(f1, 2) - (f*f2)
    x1 = x - (_n / _d)
    return a, x1


def condition(a, x):
    _, x_new = step(a, x)
    return tf.abs(x_new - x) > tf.constant(0.00001)


def get_root(a, x):
    res = tf.while_loop(condition, step, [a, x])
    return res


with tf.Session() as sess:
    a = tf.placeholder(tf.float32, shape=5)
    root = get_root(a, tf.constant(2.0))

    # get root
    result = sess.run(root, feed_dict={a: [1, 1, 1, 1, 0]})
    print(result)
