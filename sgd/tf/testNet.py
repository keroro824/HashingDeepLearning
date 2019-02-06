from __future__ import division, print_function, unicode_literals
import tensorflow as tf
from time import time
import numpy as np
# from mnist import input_data
 
# Download the dataset
# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
data = np.array([[[1,2,3,4,5]], [[5,4,3,2,1]], [[6,7,8,9,10]], [[10, 9,8,7,6]]])
label = np.array([[[1,0]], [[0, 1]], [[1,0]], [[0, 1]]])

 
def weight_variable(shape):
    # initial = tf.truncated_normal(shape, stddev=0.01)
    initial = 0.01*np.ones(shape, dtype="float32")
    return tf.Variable(initial)
 
 
def bias_variable(shape):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial)
 
 
# correct labels
y_ = tf.placeholder(tf.float32, [None, 2])
 
# input data
x = tf.placeholder(tf.float32, [None, 5])
 
# build the network
# keep_prob_input = tf.placeholder(tf.float32)
# x_drop = tf.nn.dropout(x, keep_prob=keep_prob_input)
with tf.name_scope('fc1'):
	W_fc1 = weight_variable([5, 3])
	b_fc1 = bias_variable([3])
	h_fc1 = tf.nn.relu(tf.matmul(x, W_fc1) + b_fc1)


with tf.name_scope('fc2'): 
	W_fc2 = weight_variable([3, 3])
	b_fc2 = bias_variable([3])
	h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)
 
 
with tf.name_scope('fc3'): 
	W_fc3 = weight_variable([3, 2])
	b_fc3 = bias_variable([2])
	y = tf.nn.softmax(tf.matmul(h_fc2, W_fc3) + b_fc3)
 
# define the loss function
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
 
# define training step and accuracy
train_step = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
 
# create a saver
saver = tf.train.Saver()
 
# initialize the graph
# init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(tf.global_variables_initializer())
# print(sess.run(W_fc1))
# train
batch_size = 1

start_time = time()
best_accuracy = 0.0
# print(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='fc1'))
for haha in range(10):
	for i in range(4):
	    # input_images, correct_predictions = mnist.train.next_batch(batch_size,shuffle=False)

	    input_images = data[i]
	    correct_predictions = label[i]
	    sess.run(train_step, feed_dict={x: input_images, y_: correct_predictions})
	    a = sess.run((W_fc1))
	    print("iteration " +str(i))
	    # a = sess.run(tf.gradients(cross_entropy,b_fc3),feed_dict={x: input_images, y_: correct_predictions})
	    print("layer 1")
	    for elem in a:
	    	print(elem)
	    # c = sess.run((b_fc1))
	    # print(c)

	    c = sess.run((W_fc2))
	    # b = sess.run(tf.gradients(cross_entropy,b_fc3),feed_dict={x: input_images, y_: correct_predictions})

	    print("layer 2")
	    for elem in c:
	    	print(elem)

	    b = sess.run((W_fc3))
	    # b = sess.run(tf.gradients(cross_entropy,b_fc3),feed_dict={x: input_images, y_: correct_predictions})

	    print("layer 3")
	    for elem in b:
	    	print(elem)
	    # d = sess.run((b_fc3))
	    # print(d)
	    # print([n.name for n in tf.get_default_graph().as_graph_def().node])
    

print("The training took %.4f seconds." % (time() - start_time))
 
# validate
print("Best test accuracy: %g" % best_accuracy)