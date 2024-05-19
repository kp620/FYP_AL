import pandas as pd
import numpy as np

directory = "/vol/bitbucket/kp620/FYP/dataset"
Monday_file = f'{directory}/Monday-WorkingHours.csv'
Tuesday_file = f'{directory}/Tuesday-WorkingHours.csv'
Wednesday_file = f'{directory}/Wednesday-WorkingHours.csv'
Thursday_file = f'{directory}/Thursday-WorkingHours.csv'
Friday_file = f'{directory}/Friday-WorkingHours.csv'

full_y_data = f'{directory}/y_data_iid_multiclass.csv'

# Load the data
Monday = pd.read_csv(Monday_file)
Tuesday = pd.read_csv(Tuesday_file)
Wednesday = pd.read_csv(Wednesday_file)
Thursday = pd.read_csv(Thursday_file)
Friday = pd.read_csv(Friday_file)
y = pd.read_csv(full_y_data)


Monday_length = len(Monday)
Tuesday_length = len(Tuesday)
Wednesday_length = len(Wednesday)
Thursday_length = len(Thursday)
Friday_length = len(Friday)

print("Monday length: ", Monday_length)
print("Tuesday length: ", Tuesday_length)
print("Wednesday length: ", Wednesday_length)
print("Thursday length: ", Thursday_length)
print("Friday length: ", Friday_length)


length = 0 
Monday_unique_label = np.unique(y[length : length + Monday_length])
print("Monday unique label: ", Monday_unique_label)
length += Monday_length

Tuesday_unique_label = np.unique(y[length : length + Tuesday_length])
print("Tuesday unique label: ", Tuesday_unique_label)
length += Tuesday_length

Wednesday_unique_label = np.unique(y[length : length + Wednesday_length])
print("Wednesday unique label: ", Wednesday_unique_label)
length += Wednesday_length

Thursday_unique_label = np.unique(y[length : length + Thursday_length])
print("Thursday unique label: ", Thursday_unique_label)
length += Thursday_length

Friday_unique_label = np.unique(y[length : length + Friday_length])
print("Friday unique label: ", Friday_unique_label)
length += Friday_length