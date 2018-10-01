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
    """Adds a new softmax and fully-connected layer for training.

  We need to retrain the top layer to identify our new classes, so this function
  adds the right operations to the graph, along with some variables to hold the
  weights, and then sets up all the gradients for the backward pass.

  The set up for the softmax and fully-connected layers is based on:
  https://www.tensorflow.org/versions/master/tutorials/mnist/beginners/index.html

  Args:
    class_count: Integer of how many categories of things we're trying to
    recognize.
    final_tensor_name: Name string for the new final node that produces results.
    bottleneck_tensor: The output of the main CNN graph.
    bottleneck_tensor_size: How many entries in the bottleneck vector.

  Returns:
    The tensors for the training and cross entropy results, and tensors for the
    bottleneck input and ground truth input.
  """

    layer_name = 'final_training_ops'
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            initial_value = tf.truncated_normal([bottleneck_tensor_size, class_count], stddev=0.001)
            layer_weights = tf.Variable(initial_value, name='final_weights')
        with tf.name_scope('biases'):
            layer_biases = tf.Variable(tf.zeros([class_count]), name='final_biases')
        with tf.name_scope('Wx_plus_b'):
            logits = tf.matmul(bottleneck_tensor, layer_weights) + layer_biases

    final_layer_init = tf.variables_initializer([layer_weights, layer_biases])
    final_tensor = tf.nn.softmax(logits, name=final_tensor_name)

    with tf.name_scope('cross_entropy'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(y, depth=10), logits=logits)
        with tf.name_scope('total'):
            cross_entropy_mean = tf.reduce_mean(cross_entropy)

    with tf.name_scope('train'):
        optimizer = tf.train.GradientDescentOptimizer(5e-4)
        train_step = optimizer.minimize(cross_entropy_mean, var_list=[layer_weights, layer_biases])

    return train_step, cross_entropy_mean, final_tensor, final_layer_init


with tf.Session(graph=tf.Graph()) as sess:
    tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], "model_two_layer_convnet-e05d7e52-601e-491a-a408-58e71fc796c5")
    graph = tf.get_default_graph()
    print(graph)
    bottleneck_op = graph.get_operation_by_name('AG_Bottleneck')
    print(bottleneck_op)
    bottleneck_tensor = graph.get_tensor_by_name('AG_Bottleneck:0')
    print(bottleneck_tensor)
    X_op = graph.get_operation_by_name('AG_X')
    print(X_op)
    X_tensor = graph.get_tensor_by_name('AG_X:0')
    print(X_tensor)
    y_op = graph.get_operation_by_name('AG_y')
    print(y_op)
    y_tensor = graph.get_tensor_by_name('AG_y:0')
    print(y_tensor)

    # Add the new layer that we'll be training.
    (train_step, cross_entropy, final_tensor, final_layer_init) = add_final_training_ops(10, 'final_result', y_tensor, bottleneck_tensor, 5408)

    # # Set up all our weights to their initial default values.
    # init = tf.global_variables_initializer()
    # sess.run(init)

    sess.run(final_layer_init)
    _ = sess.run([train_step], feed_dict={'AG_X:0': X_train, 'AG_y:0': y_train})

    uuid = str(uuid.uuid4())
    print(uuid)
    export_dir = "retrain_two_layer_convnet-{0}".format(uuid)
    tf.saved_model.simple_save(sess, export_dir,
                               inputs={"Input": X_tensor},
                               outputs={"Output": final_tensor})
