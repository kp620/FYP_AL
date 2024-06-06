import pandas as pd

data_dic_path = "/vol/bitbucket/kp620/FYP/dataset"
x_data = pd.read_csv(f'{data_dic_path}/x_data_iid_multiclass.csv').astype(float)
y_data = pd.read_csv(f'{data_dic_path}/y_data_iid_multiclass.csv').astype(float)

# Count each class
print(y_data.value_counts())