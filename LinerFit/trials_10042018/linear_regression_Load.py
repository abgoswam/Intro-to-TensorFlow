import tensorflow as tf
import numpy as np

export_dir = "model_linear_fit-d0966df2-a553-402d-a998-6b3753d2c3f9"

with tf.Session(graph=tf.Graph()) as sess:
    tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], export_dir)

    graph = tf.get_default_graph()
    print(graph.get_operations())
    for op in graph.get_operations():
        print(op.name)

    print("=======")
    test_X = np.asarray([6.83, 4.668, 8.9])
    pred = sess.run('AG_pred:0', feed_dict={'AG_X:0': test_X})
    print("prediction : {0}".format(pred))
