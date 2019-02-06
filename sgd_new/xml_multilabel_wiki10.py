import numpy as np
import time
from itertools import islice
from scipy.sparse import csr_matrix
import math

## Training Params
feature_dim = 101938
n_classes = 30938
hidden_dim_1 = 128
n_train = 14146
n_test = 6616
n_epochs = 10
batch_size = 256


import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import glob
train_files = glob.glob('/efs/users/beidchen/workspace/aloi/aloi_lsh/data/wiki10_shuf_train.txt')
test_files = glob.glob('/efs/users/beidchen/workspace/aloi/aloi_lsh/data/wiki10_shuf_test.txt')

def data_generator(files, batch_size, n_classes):
    while 1:
        lines = []
        for file in files:
            with open(file,'r',encoding='utf-8') as f:
                while True:
                    temp = len(lines)
                    lines += list(islice(f,batch_size-temp))
                    if len(lines)!=batch_size:
                        break
                    idxs = []
                    vals = []
                    ##
                    y_idxs = []
                    y_vals = []
                    y_batch = np.zeros([batch_size,n_classes], dtype=float)
                    count = 0
                    for line in lines:
                        itms = line.strip().split(' ')
                        ##
                        y_idxs = [int(itm) for itm in itms[0].split(',')]
                        for i in range(len(y_idxs)):
                            y_batch[count,y_idxs[i]] = 1.0/len(y_idxs)
                            # y_batch[count,y_idxs[i]] = 1.0
                        ##
                        idxs += [(count,int(itm.split(':')[0])) for itm in itms[1:]]
                        vals += [float(itm.split(':')[1]) for itm in itms[1:]]
                        count += 1
                    lines = []
                    yield (idxs, vals, y_batch)

def data_generator_tst(files, batch_size):
    while 1:
        lines = []
        for file in files:
            with open(file,'r',encoding='utf-8') as f:
                while True:
                    temp = len(lines)
                    lines += list(islice(f,batch_size-temp))
                    if len(lines)!=batch_size:
                        break
                    idxs = []
                    vals = []
                    ##
                    y_batch = [None for i in range(len(lines))]
                    count = 0
                    for line in lines:
                        itms = line.strip().split(' ')
                        ## 
                        y_batch[count] = [int(itm) for itm in itms[0].split(',')]
                        ##
                        idxs += [(count,int(itm.split(':')[0])) for itm in itms[1:]]
                        vals += [float(itm.split(':')[1]) for itm in itms[1:]]
                        count += 1
                    lines = []
                    yield (idxs, vals, y_batch)


import tensorflow as tf
x_idxs = tf.placeholder(tf.int64, shape=[None,2])
x_vals = tf.placeholder(tf.float32, shape=[None])
x = tf.SparseTensor(x_idxs, x_vals, [batch_size,feature_dim])

y = tf.placeholder(tf.float32, shape=[None,n_classes])

# with tf.device('/gpu:5'):
#     W11 = tf.Variable(tf.truncated_normal([feature_dim,hidden_dim_1//4], stddev=0.01))
#     b11 = tf.Variable(tf.zeros([hidden_dim_1//4]))
#     layer_11 = tf.nn.relu(tf.sparse_tensor_dense_matmul(x,W11)+b11)


# with tf.device('/gpu:1'):
#     W12 = tf.Variable(tf.truncated_normal([feature_dim,hidden_dim_1//4], stddev=0.01))
#     b12 = tf.Variable(tf.zeros([hidden_dim_1//4]))
#     layer_12 = tf.nn.relu(tf.sparse_tensor_dense_matmul(x,W12)+b12)


# with tf.device('/gpu:2'):
#     W13 = tf.Variable(tf.truncated_normal([feature_dim,hidden_dim_1//4], stddev=0.01))
#     b13 = tf.Variable(tf.zeros([hidden_dim_1//4]))
#     layer_13 = tf.nn.relu(tf.sparse_tensor_dense_matmul(x,W13)+b13)


# with tf.device('/gpu:3'):
#     W14 = tf.Variable(tf.truncated_normal([feature_dim,hidden_dim_1//4], stddev=0.01))
#     b14 = tf.Variable(tf.zeros([hidden_dim_1//4]))
#     layer_14 = tf.nn.relu(tf.sparse_tensor_dense_matmul(x,W14)+b14)
#     layer_1 = tf.concat([layer_11, layer_12, layer_13, layer_14], axis=-1)


W1 = tf.Variable(tf.truncated_normal([feature_dim,hidden_dim_1], stddev=2.0/math.sqrt(feature_dim+hidden_dim_1)))
b1 = tf.Variable(tf.truncated_normal([hidden_dim_1], stddev=2.0/math.sqrt(feature_dim+hidden_dim_1)))
layer_1 = tf.nn.relu(tf.sparse_tensor_dense_matmul(x,W1)+b1)

# with tf.device('/gpu:4'):
#     W1 = tf.Variable(tf.truncated_normal([hidden_dim_1,hidden_dim_2], stddev=2.0/math.sqrt(hidden_dim_1+hidden_dim_2)))
#     b1 = tf.Variable(tf.truncated_normal([hidden_dim_2], stddev=2.0/math.sqrt(hidden_dim_1+hidden_dim_2)))
#     layer_2 = tf.nn.relu(tf.matmul(layer_1,W1)+b1)


W2 = tf.Variable(tf.truncated_normal([hidden_dim_1,n_classes], stddev=2.0/math.sqrt(hidden_dim_1+n_classes)))
b2 = tf.Variable(tf.truncated_normal([n_classes], stddev=2.0/math.sqrt(n_classes+hidden_dim_1)))
logits = tf.matmul(layer_1,W2)+b2

k=1
if k==1:
    top_idxs = tf.argmax(logits, axis=1)
else:
    top_idxs = tf.nn.top_k(logits, k=k, sorted=False)[1]

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))

# loss = tf.reduce_mean(tf.losses.hinge_loss(logits=logits, labels=y))

# train_step = tf.train.AdamOptimizer().minimize(loss)


# global_step = tf.Variable(0, trainable=False)
# learning_rate = 0.1
# decay_steps = n_train//batch_size
# decay_rate = 1.0
# learning_rate = tf.train.inverse_time_decay(learning_rate, global_step,
# decay_steps, decay_rate)
# learning_rate = tf.Print(learning_rate, [learning_rate])

# train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

train_step = tf.train.AdamOptimizer(0.0001).minimize(loss)



n_steps = n_epochs*(n_train//batch_size)
training_data_generator = data_generator(train_files, batch_size, n_classes)


config = tf.ConfigProto(log_device_placement=True)
config.gpu_options.allow_growth = True

sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

import time
begin_time = time.time()

n_steps_val = n_test//batch_size
test_data_generator = data_generator_tst(test_files, batch_size)

for i in range(n_steps):
    idxs_batch, vals_batch, labels_batch = next(training_data_generator)
    sess.run(train_step, feed_dict={x_idxs:idxs_batch, x_vals:vals_batch, y:labels_batch})
    #
    if i%1000==0:
        print('Finished ',i,' steps. Time elapsed for last 100 batches = ',time.time()-begin_time)
        begin_time = time.time()
        train_loss = sess.run(loss, feed_dict={x_idxs:idxs_batch, x_vals:vals_batch, y:labels_batch})
        idxs_batch, vals_batch, labels_batch = next(test_data_generator)
        top_k_classes = sess.run(top_idxs, feed_dict={x_idxs:idxs_batch, x_vals:vals_batch})
        p_at_k = np.mean([len(np.intersect1d(top_k_classes[j],labels_batch[j]))/min(k,len(labels_batch[j])) for j in range(len(top_k_classes))])
        print('train loss: ',train_loss,' test_acc: ',p_at_k)
        #print('train loss: ',train_loss)
        print('#######################')



###################### Evaluation ####################3


num_batches = 0
p_at_k = 0

import time
begin_time = time.time()

for i in range(n_steps_val):
    idxs_batch, vals_batch, labels_batch = next(test_data_generator)
    top_k_classes = sess.run(top_idxs, feed_dict={x_idxs:idxs_batch, x_vals:vals_batch})
    p_at_k += np.mean([len(np.intersect1d(top_k_classes[j],labels_batch[j]))/min(k,len(labels_batch[j])) for j in range(len(top_k_classes))])
    #
    num_batches += 1

print('Overall p_at_1 after ',num_batches,'batches = ', p_at_k/num_batches)
print('Total time elapsed = ',time.time()-begin_time)

