import tensorflow as tf
import numpy as np

rng = np.random
training_epochs = 1000
learning_rate = 0.01
train_X = np.asarray([6.83, 4.668, 8.9, 7.91, 5.7, 8.7, 3.1, 2.1])
train_Y = (train_X * 10)


with tf.Session(graph=tf.Graph()) as sess:
    tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING],
                               "model_linear_fit-d0966df2-a553-402d-a998-6b3753d2c3f9")
    graph = tf.get_default_graph()

    # get last node
    pred = graph.get_tensor_by_name('AG_pred:0')
    X = graph.get_tensor_by_name('AG_X:0')
    Y = graph.get_tensor_by_name('AG_Y:0')
    n_samples = train_X.shape[0]

    W2 = tf.Variable(rng.randn(), name="AG_weight2")
    b2 = tf.Variable(rng.randn(), name="AG_bias2")

    # Step 1 : construct new, compute loss
    pred2 = tf.add(tf.multiply(pred, W2), b2, name="AG_pred2")
    cost = tf.reduce_sum(tf.pow(pred2 - Y, 2)) / (2 * n_samples)

    # Step 2 : one step of training
    optimizer_partial = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost, var_list=[W2, b2])
    optimizer_full = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    W1 = graph.get_tensor_by_name('AG_weight:0')
    b1 = graph.get_tensor_by_name('AG_bias:0')
    print("W1={0} b1={1}".format(sess.run(W1), sess.run(b1)))
    # print("W2={0} b2={1}".format(sess.run(W2), sess.run(b2)))

    # Initialize the variables (i.e. assign their default value)
    # init = tf.global_variables_initializer()
    init = tf.variables_initializer([W2, b2])
    sess.run(init)

    print("W1={0} b1={1}".format(sess.run(W1), sess.run(b1)))
    print("W2={0} b2={1}".format(sess.run(W2), sess.run(b2)))

    # Optimize the new layers only using optimize_partial)
    for epoch in range(training_epochs):
        for (x, y) in zip(train_X, train_Y):
            sess.run(optimizer_partial, feed_dict={X: x, Y: y})

        c = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
        print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(c), "W2=", sess.run(W2), "b2=", sess.run(b2))

    print("Optimization Finished!")
    print("W1={0} b1={1}".format(sess.run(W1), sess.run(b1)))
    print("W2=", sess.run(W2), "b2=", sess.run(b2))

    test_X = np.asarray([6.83, 4.668, 8.9])
    test_Y = sess.run(pred2, feed_dict={X: test_X})
    print("prediction : {0}".format(test_Y))

    # # Optimize the entire graph optimize_full
    # for epoch in range(training_epochs):
    #     for (x, y) in zip(train_X, train_Y):
    #         sess.run(optimizer_full, feed_dict={X: x, Y: y})
    #
    #     c = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
    #     print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(c), "W2=", sess.run(W2), "b2=", sess.run(b2))
    #
    # print("Optimization Finished!")
    # print("W1={0} b1={1}".format(sess.run(W1), sess.run(b1)))
    # print("W2=", sess.run(W2), "b2=", sess.run(b2))
    #
    # test_X = np.asarray([6.83, 4.668, 8.9])
    # test_Y = sess.run(pred2, feed_dict={X: test_X})
    # print("prediction : {0}".format(test_Y))