from __future__ import division, print_function, unicode_literals
import tensorflow as tf
from time import time
 
# Download the dataset
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
 
 
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)
 
 
def bias_variable(shape):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial)
 
 
# correct labels
y_ = tf.placeholder(tf.float32, [None, 10])
 
# input data
x = tf.placeholder(tf.float32, [None, 784])
 
# build the network
keep_prob_input = tf.placeholder(tf.float32)
x_drop = tf.nn.dropout(x, keep_prob=keep_prob_input)
 
W_fc1 = weight_variable([784, 1000])
b_fc1 = bias_variable([1000])
h_fc1 = tf.nn.relu(tf.matmul(x_drop, W_fc1) + b_fc1)
 
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
 
W_fc2 = weight_variable([1000, 1000])
b_fc2 = bias_variable([1000])
h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
 
h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)
 
 
 
W_fc3 = weight_variable([1000, 10])
b_fc3 = bias_variable([10])
y = tf.nn.softmax(tf.matmul(h_fc2_drop, W_fc3) + b_fc3)
 
# define the loss function
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
 
# define training step and accuracy
train_step = tf.train.GradientDescentOptimizer(learning_rate=1).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
 
# create a saver
saver = tf.train.Saver()
 
# initialize the graph
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
 
# train
batch_size = 1000
print("Startin Burn-In...")
saver.save(sess, 'mnist_fc_best')
# for i in range(70):
#     input_images, correct_predictions = mnist.train.next_batch(batch_size,shuffle=False)
#     if i % (60000/batch_size) == 0:
#         train_accuracy = sess.run(accuracy, feed_dict={
#             x: input_images, y_: correct_predictions, keep_prob_input: 1.0, keep_prob: 1.0})
#         print("step %d, training accuracy %g" % (i, train_accuracy))
#         # validate
#         test_accuracy = sess.run(accuracy, feed_dict={
#             x: mnist.test.images, y_: mnist.test.labels, keep_prob_input: 1.0, keep_prob: 1.0})
#         print("Validation accuracy: %g." % test_accuracy)
#     sess.run(train_step, feed_dict={x: input_images, y_: correct_predictions, keep_prob_input: 0.8, keep_prob: 0.5})
# saver.restore(sess, 'mnist_fc_best')
# print("Starting the training...")
start_time = time()
best_accuracy = 0.0
for i in range(20*60):
    input_images, correct_predictions = mnist.train.next_batch(batch_size,shuffle=False)
    if i % (60000/batch_size) == 0:
        train_accuracy = sess.run(accuracy, feed_dict={
            x: input_images, y_: correct_predictions, keep_prob_input: 1.0, keep_prob: 1.0})
        print("step %d, training accuracy %g" % (i, train_accuracy))
        # validate
        test_accuracy = sess.run(accuracy, feed_dict={
            x: mnist.test.images, y_: mnist.test.labels, keep_prob_input: 1.0, keep_prob: 1.0})
        if test_accuracy >= best_accuracy:
            saver.save(sess, 'mnist_fc_best')
            best_accuracy = test_accuracy
            print("Validation accuracy improved: %g. Saving the network." % test_accuracy)
        else:
            saver.restore(sess, 'mnist_fc_best')
            print("Validation accuracy was: %g. It was better before: %g. " % (test_accuracy, best_accuracy) +
                  "Using the old params for further optimizations.")
    sess.run(train_step, feed_dict={x: input_images, y_: correct_predictions, keep_prob_input: 1.0, keep_prob: 1.0})
print("The training took %.4f seconds." % (time() - start_time))
 
# validate
print("Best test accuracy: %g" % best_accuracy)