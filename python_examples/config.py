import numpy as np

class config:
    data_path_train = '../dataset/Amazon/amazon_train.txt'
    data_path_test = '../dataset/Amazon/amazon_test.txt'
    GPUs = '0' # empty string uses only CPU
    num_threads = 44 # Only used when GPUs is empty string
    lr = 0.0001
    ###
    feature_dim = 135909
    n_classes = 670091
    n_train = 490449
    n_test = 153025
    n_epochs = 2
    batch_size = 128
    hidden_dim = 128
    ###
    log_file = 'log_amz_ss'
    ### for sampled softmax
    n_samples = n_classes//10
    ### choose the max_labels per training sample. 
    ### If the number of true labels is < max_label,
    ### we will pad the rest of them with a dummy class (see data_generator_ss in util.py)
    max_label = 1
