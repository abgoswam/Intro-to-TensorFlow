import uuid

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

# clear old variables
tf.reset_default_graph()
X = tf.placeholder(tf.float32, [None, 32, 32, 3], name="AG_X")
y = tf.placeholder(tf.int64, [None], name="AG_y")

# setup variables
Wconv1 = tf.get_variable(name="AG_Wconv1", shape=[7, 7, 3, 32])
bconv1 = tf.get_variable(name="AG_bconv1", shape=[32])
W1 = tf.get_variable(name="AG_W1", shape=[5408, 10])
b1 = tf.get_variable(name="AG_b1", shape=[10])

# define our graph (e.g. two_layer_convnet)
a1 = tf.nn.conv2d(X, Wconv1, strides=[1, 2, 2, 1], padding='VALID') + bconv1
h1 = tf.nn.relu(a1, name="AG_ConvRelu_1")
h1_flat = tf.reshape(h1, [-1, 5408], name="AG_Bottleneck")
y_out = tf.matmul(h1_flat, W1, name="AG_FC_1") + b1

# softmax
softmax = tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(y, depth=10), logits=y_out, name="softmax")

# define our optimizer
mean_loss = tf.reduce_mean(tf.losses.hinge_loss(tf.one_hot(y, 10), logits=y_out))
optimizer = tf.train.AdamOptimizer(5e-4)  # select optimizer and set learning rate
train_step = optimizer.minimize(mean_loss)

# compute these stats
cross_entropy = tf.reduce_mean(softmax)
correct_prediction = tf.equal(tf.argmax(y_out, 1), y)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
loss_history = []
train_accuracy_history = []
val_accuracy_history = []

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(10000):
        batch = get_batch(X_train, y_train, 50)
        if i % 400 == 0:
            train_accuracy = accuracy.eval(feed_dict={X: batch[0], y: batch[1]})
            val_accuracy = accuracy.eval(feed_dict={X: X_val, y: y_val})
            train_accuracy_history.append(train_accuracy)
            val_accuracy_history.append(val_accuracy)
            print('step %d, training accuracy %g, , validation accuracy %g' % (i, train_accuracy, val_accuracy))

        _, loss_i = sess.run([train_step, cross_entropy], feed_dict={X: batch[0], y: batch[1]})
        loss_history.append(loss_i)

    print('test accuracy %g' % accuracy.eval(feed_dict={X: X_val, y: y_val}))
    uuid = str(uuid.uuid4())
    print(uuid)
    export_dir = "model_two_layer_convnet-{0}".format(uuid)
    tf.saved_model.simple_save(sess,
                               export_dir,
                               inputs={"Input": X},
                               outputs={"Output": y_out})


# Run this cell to visualize training loss and train / val accuracy
plt.subplot(2, 1, 1)
plt.title('Training loss')
plt.plot(loss_history, 'o')
plt.xlabel('Iteration')

plt.subplot(2, 1, 2)
plt.title('Accuracy')
plt.plot(train_accuracy_history, '-o', label='train')
plt.plot(val_accuracy_history, '-o', label='val')
plt.xlabel('Epoch')
plt.legend(loc='lower right')
plt.gcf().set_size_inches(15, 12)
plt.show()

for op in tf.get_default_graph().get_operations():
    print(op.name)
