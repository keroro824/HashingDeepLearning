from __future__ import division, print_function, unicode_literals
import tensorflow as tf
from time import time
import numpy as np
 
# Download the dataset
# from tensorflow.examples.tutorials.mnist import input_data
import mnist as mn
mnist = mn.read_data_sets("MNIST_data/", one_hot=True, validation_size=0)
 
 
def weight_variable(shape):
    # initial = tf.truncated_normal(shape, stddev=0.01)
    initial = 0.01*np.ones(shape, dtype="float32")
    return tf.Variable(initial)
 
 
def bias_variable(shape):
    initial = tf.constant(0.0, shape=shape, dtype="float32")
    return tf.Variable(initial)
 
 
# correct labels
y_ = tf.placeholder(tf.float32, [None, 10])
 
# input data
x = tf.placeholder(tf.float32, [None, 784])
 
# build the network
 
W_fc1 = weight_variable([784, 10])
b_fc1 = bias_variable([10])
h_fc1 = tf.nn.relu(tf.matmul(x, W_fc1) + b_fc1)
 
 
W_fc2 = weight_variable([10, 10])
b_fc2 = bias_variable([10])
h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)
 
 
 
W_fc3 = weight_variable([10, 10])
b_fc3 = bias_variable([10])
# y = tf.nn.softmax(tf.matmul(h_fc2, W_fc3) + b_fc3)
y = tf.matmul(h_fc2, W_fc3) + b_fc3
 
# define the loss function
# cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
batch_size = 10

##############TOPK##############
top_k=3
logits_new = tf.nn.softmax(y)+(y_*100000)
top_idxs = tf.nn.top_k(logits_new, k=top_k, sorted=False)[1]
#
a = tf.range(tf.shape(top_idxs)[0])
b = tf.tile(a,[top_k])
b = tf.reshape(b,[top_k, batch_size])
b = tf.transpose(b)
c = tf.stack([b,top_idxs],axis=2)
#
logits_sampled = tf.gather_nd(y,c)
y_sampled = tf.gather_nd(y_,c)
#
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits_sampled, labels=y_sampled))
# cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_sampled * tf.log(logits_sampled), reduction_indices=[1]))
# cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits_sampled, labels=y_sampled)

########
#
top_idx = tf.argmax(y, axis=1)

#############################################

 
# define training step and accuracy
train_step = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
 
# create a saver
saver = tf.train.Saver()
 
# initialize the graph
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

 
# train


start_time = time()
best_accuracy = 0.0
for i in range(6000):
    input_images, correct_predictions = mnist.train.next_batch(batch_size,shuffle=False)
    if i==5999:
        print("Hi")
        print(sess.run(W_fc3, feed_dict={x: input_images, y_: correct_predictions}))
        print(sess.run(top_idxs, feed_dict={x: input_images, y_: correct_predictions}))
        print(sess.run(logits_new, feed_dict={x: input_images, y_: correct_predictions}))
    sess.run(train_step, feed_dict={x: input_images, y_: correct_predictions})
    if i==5999:
        print(sum(input_images[0]))
        print(correct_predictions)
        a = sess.run((W_fc1))
        print("iteration " +str(i))
        # a = sess.run(tf.gradients(cross_entropy,b_fc3),feed_dict={x: input_images, y_: correct_predictions})
        print("layer 1")
        for elem in a:
            print(elem)
        # # print(a[950])
        e = sess.run((b_fc2))
        print(e)

        c = sess.run((W_fc2))

        print("layer 2")
        for elem in c:
            print(elem)
        # print (c[950])

        b = sess.run((W_fc3))
        g = sess.run(tf.gradients(cross_entropy,W_fc2),feed_dict={x: input_images, y_: correct_predictions})


        k = sess.run((b_fc3))
        print(k)

        print("layer 3")
        # for elem in g:
        #     print(elem)

        for elem in b:
            print(elem)
        # print(correct_predictions)
        # print(sess.run(top_idxs, feed_dict={x: input_images, y_: correct_predictions}))

        # print(sess.run(haha, feed_dict={x: input_images, y_: correct_predictions}))

    if i == 5999:
        train_accuracy = sess.run(accuracy, feed_dict={
            x: input_images, y_: correct_predictions})
        print("step %d, training accuracy %g" % (i, train_accuracy))
        # validate
        test_accuracy = sess.run(accuracy, feed_dict={
            x: mnist.test.images, y_: mnist.test.labels})
        if test_accuracy >= best_accuracy:
            saver.save(sess, './mnist_fc_best')
            best_accuracy = test_accuracy
            print("Validation accuracy improved: %g. Saving the network." % test_accuracy)
        else:
            saver.restore(sess, 'mnist_fc_best')
            print("Validation accuracy was: %g. It was better before: %g. " % (test_accuracy, best_accuracy) +
                  "Using the old params for further optimizations.")
print("The training took %.4f seconds." % (time() - start_time))
# validate
print("Best test accuracy: %g" % best_accuracy)