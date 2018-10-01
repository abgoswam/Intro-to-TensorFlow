import tensorflow as tf
import matplotlib.pyplot as plt
from simple_models.data_load import get_CIFAR10_data, get_batch

# Invoke the above function to get our data.
X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data()
print('Train data shape: ', X_train.shape)
print('Train labels shape: ', y_train.shape)
print('Validation data shape: ', X_val.shape)
print('Validation labels shape: ', y_val.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)

export_dir = "model_two_layer_convnet"

with tf.Session(graph=tf.Graph()) as sess:
    tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], export_dir)

    graph = tf.get_default_graph()
    print(graph.get_operations())
    for op in graph.get_operations():
        print("-----")
        print(op)
        print("-----")

    print("=======")
    a_test = sess.run('add_1:0', feed_dict={'AG_X:0': X_test})
    print("test acc : {0}".format(a_test))