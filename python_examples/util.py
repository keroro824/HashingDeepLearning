from itertools import islice
import numpy as np
from config import config

def data_generator(files, batch_size, n_classes):
    while 1:
        lines = []
        for file in files:
            with open(file,'r',encoding='utf-8') as f:
                header = f.readline()  # ignore the header
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

def data_generator_ss(files, batch_size, n_classes, max_label):
    while 1:
        lines = []
        for file in files:
            with open(file,'r',encoding='utf-8') as f:
                header = f.readline() # ignore the header
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
                        if max_label>=len(y_batch[count]): # 
                            y_batch[count] += [n_classes for i in range(max_label-len(y_batch[count]))]
                        else:
                            y_batch[count] = np.random.choice(y_batch[count], max_label, replace=False)
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
                header = f.readline() # ignore the header
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

