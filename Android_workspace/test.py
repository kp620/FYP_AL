import numpy as np

file = '/vol/bitbucket/kp620/FYP/Android_workspace/data/gen_apigraph_drebin/2013-01_selected.npz'


data = np.load(file)
x_train = data['X_train']
y_train = data['y_train']
y_train = np.where(y_train == 0, 0, 1) # Convert to binary

#count 1 and 0
print('Number of benign apps:', np.sum(y_train == 0))
print('Number of malicious apps:', np.sum(y_train == 1))