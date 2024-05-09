import numpy as np
import os
import pickle
import subprocess

directory = '/vol/bitbucket/kp620/FYP/Android_workspace/data/gen_androzoo_drebin'


def load_data(file):
    data = np.load(file)
    x_train = data['X_train']
    y_train = data['y_train']
    y_mal_family = data['y_mal_family']
    return x_train, y_train, y_mal_family

def main(directory):
    x_train = []
    y_train = []
    y_mal_family = []
    for file in os.listdir(directory):
        if file.endswith('.npz'):
            command = "echo 'Loading file: " + file + "'"
            subprocess.call(command, shell=True)
            x, y, y_mal = load_data(os.path.join(directory, file))
            x_train.append(x)
            y_train.append(y)
            y_mal_family.append(y_mal)
    x_train = np.concatenate(x_train)
    y_train = np.concatenate(y_train)
    y_mal_family = np.concatenate(y_mal_family)
    return x_train, y_train, y_mal_family