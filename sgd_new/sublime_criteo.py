import numpy as np
import time
from itertools import islice
from scipy.sparse import csr_matrix
import math

## Training Params
feature_dim = 1000000
n_classes = 2
hidden_dim1 = 1024
hidden_dims = [1024, 1024]
n_train = 45840617
n_test = 6042135
n_epochs = 1
batch_size = 512


import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="5,6,7"

import glob
test_files = glob.glob('/efs/users/beidchen/workspace/aloi/aloi_lsh/data/criteo.kaggle2014.test.svm')


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

#load weights
weights = np.load('/efs/users/beidchen/workspace/SUBLIME/HashingDeepLearning/sgd_new/log_new/criteo/weight_criteo_sim_4_5_e-4_B512.npz')

Weight1_1 = weights['w_layer_0'].T[:, :hidden_dim1//2]
Weight1_2 = weights['w_layer_0'].T[:,hidden_dim1//2:]
Bias1_1 = weights['b_layer_0'].T[:hidden_dim1//2]
Bias1_2 = weights['b_layer_0'].T[hidden_dim1//2:]
Weight_2 = weights['w_layer_3'].T
Bias_2 = weights['b_layer_3'].T


import tensorflow as tf
x_idxs = tf.placeholder(tf.int64, shape=[None,2])
x_vals = tf.placeholder(tf.float32, shape=[None])
x = tf.SparseTensor(x_idxs, x_vals, [batch_size,feature_dim])

y = tf.placeholder(tf.float32, shape=[None,n_classes])

with tf.device('/gpu:2'):
    W11 = tf.Variable(Weight1_1)
    b11 = tf.Variable(Bias1_1)
    layer_11 = tf.nn.relu(tf.sparse_tensor_dense_matmul(x,W11)+b11)


with tf.device('/gpu:1'):
    W12 = tf.Variable(Weight1_2)
    b12 = tf.Variable(Bias1_2)
    layer_12 = tf.nn.relu(tf.sparse_tensor_dense_matmul(x,W12)+b12)
    layer_1 = tf.concat([layer_11, layer_12], axis=-1)


layer = 0
W = [None for i in range(len(hidden_dims))]
b = [None for i in range(len(hidden_dims))]

for hidden_dim in hidden_dims:
# with tf.device('/gpu:4'):
    W[layer] = tf.Variable(weights['w_layer_'+str(layer+1)].T)
    b[layer] = tf.Variable(weights['b_layer_'+str(layer+1)].T)
    layer_2 = tf.nn.relu(tf.matmul(layer_1,W[layer])+b[layer])
    hidden_dim1 = hidden_dim
    layer_1 = layer_2
    layer+=1

W2 = tf.Variable(Weight_2)
b2 = tf.Variable(Bias_2)
logits = tf.matmul(layer_1,W2)+b2

k=1
if k==1:
    top_idxs = tf.argmax(logits, axis=1)
else:
    top_idxs = tf.nn.top_k(logits, k=k, sorted=False)[1]

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))

train_step = tf.train.AdamOptimizer(0.0001).minimize(loss)





# input_dic = {W11:weights['w_layer_0'].T[:, :hidden_dim1//2], W12:weights['w_layer_0'].T[:,hidden_dim1//2:], b11:weights['b_layer_0'].T[:hidden_dim1//2], b12:weights['b_layer_0'].T[hidden_dim1//2:], W2:weights['w_layer_3'].T, b2:weights['b_layer_3'].T}
# for i in range(len(hidden_dims)):
#     input_dic[W[i]] = weights['w_layer_'+str(i+1)].T
#     input_dic[b[i]] = weights['b_layer_'+str(i+1)].T



config = tf.ConfigProto(log_device_placement=True)
config.gpu_options.allow_growth = True

sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

###################### Evaluation ####################3


probs = tf.nn.softmax(logits)

num_batches = 0

n_steps_val = n_test//batch_size
test_data_generator = data_generator_tst(test_files, batch_size)

import time
begin_time = time.time()
title = 'Id,Predicted'
with open("/efs/users/beidchen/workspace/submission_epoch2.csv", 'w') as out:
    out.write('{0}\n'.format(title))
    for i in range(n_steps_val):
        idxs_batch, vals_batch, labels_batch = next(test_data_generator)
        scores = sess.run(probs, feed_dict={x_idxs:idxs_batch, x_vals:vals_batch})
        indices = np.array(range(60000000+i*batch_size, 60000000+(i+1)*batch_size))
        for elem in zip(indices.astype(str), scores[:,1].astype(str)):
            print(','.join(elem), file=out)
    for i in range(66042112, 66042135):
        print(str(i)+",0.0", file=out)



