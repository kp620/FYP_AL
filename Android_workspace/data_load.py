import numpy as np
import os
import subprocess

directory = '/vol/bitbucket/kp620/FYP/Android_workspace/data/gen_androzoo_drebin'

# command = "echo 'Loading data...'"
# subprocess.call(command, shell=True)
# x_train = np.load(f'{directory}/x_train.npz')['x_train']
# command = "echo 'x_train loaded'"
# subprocess.call(command, shell=True)
y_train = np.load(f'{directory}/y_train.npz')['y_train']
command = "echo 'y_train loaded'"
subprocess.call(command, shell=True)
# y_mal_family = np.load(f'{directory}/y_mal_family.npz')['y_mal_family']
# command = "echo 'y_mal_family loaded'"
# subprocess.call(command, shell=True)

unique_labels = np.unique(y_train)
print(len(unique_labels))