import numpy as np
import time
from itertools import islice
from scipy.sparse import csr_matrix

## Training Params
# feature_dim = 1617899
# n_classes = 325056
# hidden_dim_1 = 100
# n_train = 1778351
# n_test = 587084
# n_epochs = 1
# batch_size = 100

feature_dim = 782585
n_classes = 205443
hidden_dim_1 = 128
n_train = 196606
n_test = 100095
n_epochs = 10
batch_size = 256
# NUM_THREADS = 32

# feature_dim = 135909
# n_classes = 670091
# hidden_dim_1 = 128
# n_train = 490449
# n_test = 153025
# n_epochs = 1
# batch_size = 100


import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import glob
# train_files = glob.glob('/search_labs/users/medinit/SUBLIME/mach/Amazon-3M/amazon-3M_train_shuf')
# test_files = glob.glob('/search_labs/users/medinit/SUBLIME/mach/Amazon-3M/amazon-3M_test_shuf')
train_files = glob.glob('/beidi/NN/data/deliciousLarge_shuf_train.txt')
test_files = glob.glob('/beidi/NN/data/deliciousLarge_shuf_test.txt')
weights = np.load('/beidi/NN/HashingDeepLearning/bgd_new/log_paper/delicious/weightsave.npz')


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



Weight_1 = weights['w_layer_0'].T
Bias_1 = weights['b_layer_0'].T
Weight_2 = weights['w_layer_1'].T
Bias_2 = weights['b_layer_1'].T

import tensorflow as tf
x_idxs = tf.placeholder(tf.int64, shape=[None,2])
x_vals = tf.placeholder(tf.float32, shape=[None])
x = tf.SparseTensor(x_idxs, x_vals, [batch_size,feature_dim])

y = tf.placeholder(tf.float32, shape=[None,n_classes])


# W1 = tf.Variable(tf.truncated_normal([feature_dim,hidden_dim_1], stddev=0.01))
# b1 = tf.Variable(tf.zeros([hidden_dim_1]))
W1 = tf.Variable(Weight_1)
b1 = tf.Variable(Bias_1)
layer_1 = tf.nn.relu(tf.sparse_tensor_dense_matmul(x,W1)+b1)


# W2 = tf.Variable(tf.truncated_normal([hidden_dim_1,n_classes], stddev=0.01))
# b2 = tf.Variable(tf.zeros([n_classes]))
W2 = tf.Variable(Weight_2)
b2 = tf.Variable(Bias_2)
logits = tf.matmul(layer_1,W2)+b2

k=1
if k==1:
    top_idxs = tf.argmax(logits, axis=1)
else:
    top_idxs = tf.nn.top_k(logits, k=k, sorted=False)[1]

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))

# loss = tf.reduce_mean(tf.losses.hinge_loss(logits=logits, labels=y))

train_step = tf.train.AdamOptimizer(0.00001).minimize(loss)


# global_step = tf.Variable(0, trainable=False)
# learning_rate = 0.1
# decay_steps = 1.0
# decay_rate = 1.0
# learning_rate = tf.train.inverse_time_decay(learning_rate, global_step,
# decay_steps, decay_rate)
# learning_rate = tf.Print(learning_rate, [learning_rate])

# train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)




n_steps = n_epochs*(n_train//batch_size)
training_data_generator = data_generator(train_files, batch_size, n_classes)


# config = tf.ConfigProto(inter_op_parallelism_threads=NUM_THREADS, intra_op_parallelism_threads=NUM_THREADS)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

import time
begin_time = time.time()
total_time = 0

with open("/beidi/NN/HashingDeepLearning/bgd_new/log_paper/delicious/log_deli_tf_44", 'w') as out:
    for i in range(n_steps):
        if i%100==0:
            total_time+=time.time()-begin_time
            print('Finished ',i,' steps. Time elapsed for last 100 batches = ',time.time()-begin_time)
            n_steps_val = n_test//batch_size
            test_data_generator = data_generator_tst(test_files, batch_size)
            tmp_k = 0
            for h in range(20):
                idxs_batch, vals_batch, labels_batch = next(test_data_generator)
                top_k_classes = sess.run(top_idxs, feed_dict={x_idxs:idxs_batch, x_vals:vals_batch})
                tmp_k += np.mean([len(np.intersect1d(top_k_classes[j],labels_batch[j]))/min(k,len(labels_batch[j])) for j in range(len(top_k_classes))])
            print(' test_acc: ',tmp_k/20)
            #print('train loss: ',train_loss)
            print('#######################')
            print(i,int(total_time),tmp_k/20 , file=out)
            begin_time = time.time()
        idxs_batch, vals_batch, labels_batch = next(training_data_generator)
        sess.run(train_step, feed_dict={x_idxs:idxs_batch, x_vals:vals_batch, y:labels_batch})
        if i%(n_train//batch_size)==(n_train//batch_size-1):
            total_time+=time.time()-begin_time
            print('Finished ',i,' steps. Time elapsed for last 100 batches = ',time.time()-begin_time)
            n_steps_val = n_test//batch_size
            test_data_generator = data_generator_tst(test_files, batch_size)
            num_batches = 0
            p_at_k = 0
            for l in range(n_steps_val):
                idxs_batch, vals_batch, labels_batch = next(test_data_generator)
                top_k_classes = sess.run(top_idxs, feed_dict={x_idxs:idxs_batch, x_vals:vals_batch})
                p_at_k += np.mean([len(np.intersect1d(top_k_classes[j],labels_batch[j]))/min(k,len(labels_batch[j])) for j in range(len(top_k_classes))])
                num_batches += 1
            print('Overall p_at_1 after ',num_batches,'batches = ', p_at_k/num_batches)
            print(i, int(total_time) ,p_at_k/num_batches , file=out)
            begin_time = time.time()


# n_steps_val = n_test//batch_size
# test_data_generator = data_generator_tst(test_files, batch_size)
# num_batches = 0
# p_at_k = 0

# for l in range(n_steps_val):
#     idxs_batch, vals_batch, labels_batch = next(test_data_generator)
#     top_k_classes = sess.run(top_idxs, feed_dict={x_idxs:idxs_batch, x_vals:vals_batch})
#     p_at_k += np.mean([len(np.intersect1d(top_k_classes[j],labels_batch[j]))/min(k,len(labels_batch[j])) for j in range(len(top_k_classes))])
#     num_batches += 1

    
# print('Overall p_at_1 after ',n_steps_val,'batches = ', p_at_k/n_steps_val)
# begin_time = time.time()

