import tensorflow as tf

export_dir = "model_two_layer_convnet-e05d7e52-601e-491a-a408-58e71fc796c5"
with tf.Session(graph=tf.Graph()) as sess:
    tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], export_dir)

    # # Desired variable is called "AG_bconv1:0".
    # var = [v for v in tf.trainable_variables() if v.name == "AG_bconv1:0"][0]
    # print(sess.run(var))

    graph = tf.get_default_graph()
    print(graph)
    bconv1_tensor = graph.get_tensor_by_name('AG_bconv1:0')
    print(bconv1_tensor)
    print(sess.run(bconv1_tensor))

export_dir = "retrain_two_layer_convnet-4c267981-e9da-41ef-b33c-713d55056f73"
with tf.Session(graph=tf.Graph()) as sess:
    tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], export_dir)

    # # Desired variable is called "AG_bconv1:0".
    # var = [v for v in tf.trainable_variables() if v.name == "AG_bconv1:0"][0]
    # print(sess.run(var))

    graph = tf.get_default_graph()
    print(graph)
    bconv1_tensor = graph.get_tensor_by_name('AG_bconv1:0')
    print(bconv1_tensor)
    print(sess.run(bconv1_tensor))
    # for v in tf.trainable_variables():
    #     print(v.name)
    final_biases_tensor = graph.get_tensor_by_name('final_training_ops/biases/final_biases:0')
    print(final_biases_tensor)
    print(sess.run(final_biases_tensor))

export_dir = "retrain_two_layer_convnet-7b375fc3-8ca5-41c9-b8fc-4cf784b78be8"
with tf.Session(graph=tf.Graph()) as sess:
    tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], export_dir)

    # # Desired variable is called "AG_bconv1:0".
    # var = [v for v in tf.trainable_variables() if v.name == "AG_bconv1:0"][0]
    # print(sess.run(var))

    graph = tf.get_default_graph()
    print(graph)
    bconv1_tensor = graph.get_tensor_by_name('AG_bconv1:0')
    print(bconv1_tensor)
    print(sess.run(bconv1_tensor))
    # for v in tf.trainable_variables():
    #     print(v.name)
    final_biases_tensor = graph.get_tensor_by_name('final_training_ops/biases/final_biases:0')
    print(final_biases_tensor)
    print(sess.run(final_biases_tensor))
