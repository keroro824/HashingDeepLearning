import csv
from itertools import islice
import numpy as np
import random

lines = []
n_classes = 10
n_train = 60000
n_test = 10000
batch_size = 60000

with open("/Users/beidchen/Documents/work/SUBLIME/HashingDeepLearning/bgd/mnist_multi.scale", "w") as w:
	with open("/Users/beidchen/Documents/work/SUBLIME/HashingDeepLearning/bgd/mnist.scale") as f:
		lines = f.readlines()
		print len(lines)
		for line in lines:
			itms = line.split(' ')
			y_idxs = [itm for itm in itms[0].split(',')]
			extra_class = random.randint(0, 9)
			if str(extra_class) not in y_idxs:
				y_idxs.append(str(extra_class))
			newline = ','.join(y_idxs)
			newline +=' '
			newline += ' '.join(itms[1:])
			w.write(newline)

    # while True:
	   #  temp = len(lines)
	   #  lines += list(islice(f,batch_size-temp))
	   #  if len(lines)!=batch_size:
	   #      break
	   #  idxs = []
	   #  vals = []
	   #  ##
	   #  y_idxs = []
	   #  y_vals = []
	   #  y_batch = np.zeros([batch_size,n_classes], dtype=float)
	   #  count = 0
	   #  for line in lines:
	   #      itms = line.strip().split(' ')
	   #      ##
	   #      y_idxs = [int(itm) for itm in itms[0].split(',')]
	   #      extra_class = random.randint(0, 10)
	   #      if extra_class not in y_idxs:
	   #      	y_idxs.append(extra_class)
	   #      for i in range(len(y_idxs)):
	   #          y_batch[count,y_idxs[i]] = 1.0/len(y_idxs)
	   #      ##
	   #      idxs += [(count,int(itm.split(':')[0])) for itm in itms[1:]]
	   #      vals += [float(itm.split(':')[1]) for itm in itms[1:]]
	   #      count += 1
	   #  lines = []
	    

