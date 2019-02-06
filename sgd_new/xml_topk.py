import numpy as np
import time
from itertools import islice
from scipy.sparse import csr_matrix
import tensorflow as tf

## Training Params
feature_dim = 784
n_classes = 10
hidden_dim_1 = 10
hidden_dim_2 = 10
n_train = 60000
n_test = 10000
n_epochs = 1
batch_size = 100
# feature_dim = 1617899
# n_classes = 325056
# hidden_dim_1 = 256
# n_train = 1778351
# n_test = 587084
# n_epochs = 1
# batch_size = 10

top_k = int(n_classes*0.4)

import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="1"

import glob
# train_files = glob.glob('/search_labs/users/medinit/SUBLIME/mach/Amazon-3M/amazon-3M_train_shuf')
# test_files = glob.glob('/search_labs/users/medinit/SUBLIME/mach/Amazon-3M/amazon-3M_test_shuf')
# train_files = glob.glob('/search_labs/users/medinit/SUBLIME/mach/wiki/wikiLSHTC_shuf_train.txt')
# test_files = glob.glob('/search_labs/users/medinit/SUBLIME/mach/wiki/wikiLSHTC_shuf_test.txt')
train_files = glob.glob('/Users/beidchen/Documents/work/SUBLIME/HashingDeepLearning/bgd/mnist_multi.scale')
test_files = glob.glob('/Users/beidchen/Documents/work/SUBLIME/HashingDeepLearning/bgd/mnist_multi.scale.t')

def data_generator(files, batch_size, n_classes):
    while 1:
        lines = []
        for file in files:
            with open(file,'r') as f:
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
            with open(file,'r') as f:
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



x_idxs = tf.placeholder(tf.int64, shape=[None,2])
x_vals = tf.placeholder(tf.float32, shape=[None])
x = tf.SparseTensor(x_idxs, x_vals, [batch_size,feature_dim])
#
y = tf.placeholder(tf.float32, shape=[None,n_classes])
#
# W1 = tf.Variable(tf.ones([feature_dim,hidden_dim_1])*0.05)

params = np.load('/Users/beidchen/Documents/work/SUBLIME/HashingDeepLearning/sgd_new/weights.npz')

# W1 = tf.Variable(tf.truncated_normal([feature_dim,hidden_dim_1], stddev=0.01))
# b1 = tf.Variable(tf.zeros([hidden_dim_1]))
W1 = tf.Variable(params['weight_0'])
b1 = tf.Variable(params['bias_0'])
layer_1 = tf.nn.relu(tf.sparse_tensor_dense_matmul(x,W1)+b1)
#
# W2 = tf.Variable(tf.ones([hidden_dim_1,hidden_dim_2])*0.05)
# W2 = tf.Variable(tf.truncated_normal([hidden_dim_1,hidden_dim_2], stddev=0.01))
# b2 = tf.Variable(tf.zeros([hidden_dim_2]))
W2 = tf.Variable(params['weight_1'])
b2 = tf.Variable(params['bias_1'])
layer_2 = tf.nn.relu(tf.matmul(layer_1,W2)+b2)
###### Comment next  4 lines for top-k softmax
# W2 = tf.Variable(tf.truncated_normal([hidden_dim_1,n_classes], stddev=0.05))
# b2 = tf.Variable(tf.truncated_normal([n_classes], stddev=0.05))
# logits = tf.matmul(layer_1,W2)+b2
# loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
#
################### Top-k
#
# W3 = tf.Variable(tf.ones([n_classes,hidden_dim_2])*0.05)
W3 = tf.Variable(params['weight_2'])
b3 = tf.Variable(params['bias_2'])
# W3 = tf.Variable(tf.truncated_normal([n_classes,hidden_dim_2], stddev=0.01))
# b3 = tf.Variable(tf.zeros([n_classes]))
#
logits = tf.matmul(layer_2, tf.transpose(W3))
logits = tf.nn.bias_add(logits, b3)
#
eval_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits))
#
logits_new = logits+(y*100000)
top_idxs = tf.nn.top_k(logits_new, k=top_k, sorted=True)[1]
#
a = tf.range(tf.shape(top_idxs)[0])
b = tf.tile(a,[top_k])
b = tf.reshape(b,[top_k, batch_size])
b = tf.transpose(b)
c = tf.stack([b,top_idxs],axis=2)
#
logits_sampled = tf.gather_nd(logits,c)
y_sampled = tf.gather_nd(y,c)
#
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits_sampled, labels=y_sampled))

delta = tf.nn.softmax_cross_entropy_with_logits(logits=logits_sampled, labels=y_sampled)
# delta = tf.nn.softmax(logits_sampled)-y_sampled
########
#
k=1
if k==1:
    top_idx = tf.argmax(logits, axis=1)
else:
    top_idx = tf.nn.top_k(logits, k=k, sorted=False)[1]


optimizer = tf.train.AdamOptimizer(0.01)
grad = optimizer.compute_gradients(loss)
train_step = optimizer.minimize(loss)


#
n_steps = n_epochs*(n_train//batch_size)
training_data_generator = data_generator(train_files, batch_size, n_classes)
#
sess = tf.Session()
sess.run(tf.global_variables_initializer())

n_steps_val = n_test//batch_size
test_data_generator = data_generator_tst(test_files, batch_size)


# np.savez_compressed('weights.npz', weight_0=sess.run(W1), bias_0=sess.run(b1), weight_1=sess.run(W2), bias_1=sess.run(b2), weight_2=sess.run(W3).transpose(), bias_2=sess.run(b3))


import time
begin_time = time.time()
for i in range(n_steps):
    idxs_batch, vals_batch, labels_batch = next(training_data_generator)
    if i==599:
        print(sess.run(top_idxs, feed_dict={x_idxs:idxs_batch, x_vals:vals_batch, y:labels_batch}))
        print(sess.run(logits, feed_dict={x_idxs:idxs_batch, x_vals:vals_batch, y:labels_batch}))

        actives = sess.run(top_idxs, feed_dict={x_idxs:idxs_batch, x_vals:vals_batch, y:labels_batch})
        # actives = np.sort(actives)
        np.savetxt("actives1", actives.astype(int), fmt='%d', delimiter="")
        # print(sess.run(delta, feed_dict={x_idxs:idxs_batch, x_vals:vals_batch, y:labels_batch}))
        # print(sess.run((W3)))
        # print(sess.run(b3))
        for gv in grad:
            print(gv[1].name)
            print(sess.run(gv[0],feed_dict={x_idxs:idxs_batch, x_vals:vals_batch, y:labels_batch}))

    sess.run(train_step, feed_dict={x_idxs:idxs_batch, x_vals:vals_batch, y:labels_batch})
    #
    if i==599:
        # print('Finished ',i,' steps. Time elapsed for last 100 batches = ',time.time()-begin_time)
        # begin_time = time.time()
        # train_loss = sess.run(loss, feed_dict={x_idxs:idxs_batch, x_vals:vals_batch, y:labels_batch})
        # idxs_batch, vals_batch, labels_batch = next(test_data_generator)
        # top_k_classes = sess.run(top_idx, feed_dict={x_idxs:idxs_batch, x_vals:vals_batch})
        # p_at_k = np.mean([len(np.intersect1d(top_k_classes[j],labels_batch[j]))/min(k,len(labels_batch[j])) for j in range(len(top_k_classes))])
        # print('train loss: ',train_loss,' test_acc: ',p_at_k)
        # #print('train loss: ',train_loss)
        # print('#######################')


        print(sess.run((W3)))
        print(sess.run(b3))



###################### Evaluation ####################3

num_batches = 0
p_at_k = 0

import time
begin_time = time.time()

for i in range(n_steps_val):
    idxs_batch, vals_batch, labels_batch = next(test_data_generator)
    top_k_classes = sess.run(top_idx, feed_dict={x_idxs:idxs_batch, x_vals:vals_batch})
    p_at_k += np.mean([len(np.intersect1d(top_k_classes[j],labels_batch[j]))/min(k,len(labels_batch[j])) for j in range(len(top_k_classes))])
    #
    num_batches += 1

print('Overall p_at_1 after ',num_batches,'batches = ', p_at_k/num_batches)
print('Total time elapsed = ',time.time()-begin_time)

