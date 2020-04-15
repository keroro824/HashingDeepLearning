#!/usr/bin/python3

import numpy as np

print("Starting")

data = np.load('../savedWeight.npz')

print(data["w_layer_0"])
print(data["b_layer_0"])

print("Finished")
