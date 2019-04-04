import numpy as np

class config:
    data_path_train = '../dataset/Amazon/amazon_train.txt'
    data_path_test = '../dataset/Amazon/amazon_test.txt'
    data_path = '../dataset/Amazon/amazon_train.txt'
    GPUs = '' # empty string uses only CPU
    num_threads = 96 # Only used when GPUs is empty string
    lr = 0.0001
    ###
    feature_dim = 135909
    n_classes = 670091
    n_train = 490449
    n_test = 153025
    n_epochs = 20
    batch_size = 128
    hidden_dim = 128
    ###
    log_file = 'log'

    ### for sampled softmax
    n_samples = 670091//10
    max_label = 100
