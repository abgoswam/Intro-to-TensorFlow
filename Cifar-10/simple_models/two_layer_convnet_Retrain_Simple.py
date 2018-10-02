import tensorflow as tf
from tensorflow.python.platform import gfile
import matplotlib.pyplot as plt
from simple_models.data_load import get_CIFAR10_data, get_batch
import uuid

# Invoke the above function to get our data.
X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data()
print('Train data shape: ', X_train.shape)
print('Train labels shape: ', y_train.shape)
print('Validation data shape: ', X_val.shape)
print('Validation labels shape: ', y_val.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)

batch = get_batch(X_train, y_train, 50)


def add_final_training_ops(class_count, final_tensor_name, y, bottleneck_tensor, bottleneck_tensor_size):
    # Step 1 : get penultimate node, replace with new FC layer, compute loss
    initial_value = tf.truncated_normal([bottleneck_tensor_size, class_count], stddev=0.001)
    layer_weights = tf.Variable(initial_value, name='final_weights')
    layer_biases = tf.Variable(tf.zeros([class_count]), name='final_biases')
    logits = tf.matmul(bottleneck_tensor, layer_weights) + layer_biases

    final_layer_init = tf.variables_initializer([layer_weights, layer_biases])
    final_tensor = tf.nn.softmax(logits, name=final_tensor_name)

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(y, depth=10), logits=logits)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    # Step 2: one step of re-training
    optimizer = tf.train.GradientDescentOptimizer(5e-4)
    train_step = optimizer.minimize(cross_entropy_mean, var_list=[layer_weights, layer_biases])

    return train_step, cross_entropy_mean, final_tensor, final_layer_init


with tf.Session(graph=tf.Graph()) as sess:
    tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], "model_two_layer_convnet-e05d7e52-601e-491a-a408-58e71fc796c5")
    graph = tf.get_default_graph()

    # get penultimate node
    bottleneck_tensor = graph.get_tensor_by_name('AG_Bottleneck:0')

    X_tensor = graph.get_tensor_by_name('AG_X:0')
    y_tensor = graph.get_tensor_by_name('AG_y:0')

    # Add the new layer that we'll be training.
    (train_step, cross_entropy, final_tensor, final_layer_init) = add_final_training_ops(10, 'final_result', y_tensor, bottleneck_tensor, 5408)

    sess.run(final_layer_init)
    _ = sess.run([train_step], feed_dict={'AG_X:0': X_train, 'AG_y:0': y_train})

