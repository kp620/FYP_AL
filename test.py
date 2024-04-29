import numpy as np

data_dic_path = "/vol/bitbucket/kp620/FYP/dataset"
selected_indice = np.load(f'{data_dic_path}/selected_indice.npy')
not_selected_indice = np.load(f'{data_dic_path}/not_selected_indice.npy')

print(len(selected_indice))

print(len(not_selected_indice))