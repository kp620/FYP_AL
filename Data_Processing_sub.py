import requests
import zipfile
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

class GetData():
    def __init__(self, label_mode, url, download_path, extract_to_directory):
        self.label_mode = label_mode
        self.config = {
            "ids17_url": url,
            "download_path": download_path,
            "extract_to_directory": extract_to_directory
        }
    
    def download_and_unzip_dataset(self):
        url = self.config['ids17_url']
        download_path = self.config['download_path']
        extract_to = self.config['extract_to_directory']

        with requests.get(url, stream=True) as response:
            if response.status_code == 200:
                total_size = int(response.headers.get('content-length', 0))
                block_size = 1024 # 1 Kilobyte
                print("Start downloading...")

                with open(download_path, "wb") as file:
                    for data in response.iter_content(block_size):
                        file.write(data)
                        progress = file.tell()
                        percent_complete = 100 * progress / total_size
                        print(f"\rDownloaded {percent_complete:.2f}%", end="")

                print("\nDownload completed successfully.")
                if os.path.exists(download_path):
                    print(f"Unzipping '{download_path}'...")
                    with zipfile.ZipFile(download_path, 'r') as zip_ref:
                        zip_ref.extractall(extract_to)
                        print(f"Unzipped successfully into '{extract_to}'")
                    os.remove(download_path)
                    print(f"Deleted ZIP file '{download_path}'")
            else:
                print("Failed to download the file. Status code:", response.status_code)
                return
            
    def set_pandas_options(self):
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)

    def read_and_clean_data(self,file_path):
        print(f'Reading and Cleaning Data: {file_path}')
        df = pd.read_csv(file_path)
        # Remove duplicate rows
        df = df.drop_duplicates().dropna()
        return df

    def combine_dataframes(self,dfs):
        combined_df = pd.concat(dfs, ignore_index=True)
        return combined_df
    
    def label_data(self,data):
        print(f'Labeling The Data: {self.label_mode}')
        if self.label_mode == "binary":
            # Either BENIGN or MALICIOUS
            data["Label"] = data["Label"].apply(lambda x: 0 if x == "BENIGN" else 1)
        elif self.label_mode == "multiclass":
            # Extracts all the unique values from the Label column
            unique_labels = data['Label'].unique().tolist()
            if 'BENIGN' in unique_labels:
                unique_labels.remove('BENIGN')
            # Each unique label (except for 'BENIGN') is assigned a unique integer value, starting from 1
            label_mapping = {label: index + 1 for index, label in enumerate(unique_labels)}
            label_mapping['BENIGN'] = 0
            data['Label'] = data['Label'].map(label_mapping)
        return data
    
    def process_data(self,df):
        # Change all Attempted flow to BENIGN
        attempted_labels = [s for s in df['Label'].unique() if 'Attempted' in s]
        df.replace(attempted_labels, 'BENIGN', inplace=True)
        # Avoid Shortcut learning 
        cols_to_remove = ['Flow ID', 'Src IP', 'Src Port', 'Dst IP', 'Timestamp']
        df = df.drop(cols_to_remove, axis=1)
        return df


    def prepare_data(self):
        self.set_pandas_options()
        print('Processing the Data')
        ds_dic = self.config['extract_to_directory']
        file_paths = [f"{ds_dic}/Monday-WorkingHours.csv", f"{ds_dic}/Tuesday-WorkingHours.csv", f"{ds_dic}/Wednesday-WorkingHours.csv", f"{ds_dic}/Thursday-WorkingHours.csv", f"{ds_dic}/Friday-WorkingHours.csv"]
        print(file_paths)
        dataframes = [self.read_and_clean_data(file_path) for file_path in file_paths]
        for i, df in enumerate(dataframes, 1):
            print(f'Day {i}: {len(df)} rows')
        # Stack vertically
        combined_df = self.combine_dataframes(dataframes)

        # Process the combined dataframe
        processed_df = self.process_data(combined_df)
        nRow, nCol = processed_df.shape
        print(f'Processed data has {nRow} rows and {nCol} columns')
        processed_df = self.label_data(processed_df)
        processed_df.to_csv(f'{ds_dic}/clean_data.csv', index=False)
        print(f' Data Saved: clean_data.csv')

class SplitData():
    def __init__(self, label_mode, url, download_path, extract_to_directory):
        self.label_mode = label_mode
        self.config = {
            "ids17_url": url,
            "download_path": download_path,
            "extract_to_directory": extract_to_directory
        }
        self.ds_dic = self.config['extract_to_directory']
        self.data = pd.read_csv(f'{self.ds_dic}/clean_data.csv')
        self.data = self.data.astype(float)
        self.length = len(self.data)

    def time_aware_split(self):
        print('Total number of data: ', self.length)
        print('Split and Scale Data')
        print('Start Time Aware Split')
        time_aware = 693682
        train_data = self.data[:time_aware]
        train_data_length = len(train_data)
        print('Length of training data: ', train_data_length)
        y_train, x_train = train_data["Label"], train_data.drop('Label', axis=1)
        test_data = self.data[time_aware:]
        test_data_length = len(test_data)
        print('Length of training data: ', test_data_length)
        y_test, x_test = test_data["Label"], test_data.drop('Label', axis=1)
        # Replace infinite value with NaN
        x_test.replace([np.inf, -np.inf], np.nan, inplace=True)
        # By default, np.nan_to_num() replaces NaN with zero (0) and also converts positive and negative infinity to the largest or smallest finite floating point values representable by x_test's dtype, respectively. 
        x_test = np.nan_to_num(x_test)
        x_train.replace([np.inf, -np.inf], np.nan, inplace=True)
        x_train = np.nan_to_num(x_train)

        # min-max normalization to scale feature data
        scaler = MinMaxScaler().fit(x_train)
        x_train, x_test = scaler.transform(x_train), scaler.transform(x_test)
        x_train_df = pd.DataFrame(x_train)
        x_test_df = pd.DataFrame(x_test)
        x_train_df.to_csv(f'{self.ds_dic}/x_train_time_aware.csv', index=False)
        x_test_df.to_csv(f'{self.ds_dic}/x_test_time_aware.csv', index=False)
        y_train.to_csv(f'{self.ds_dic}/y_train_time_aware.csv', index=False)
        y_test.to_csv(f'{self.ds_dic}/y_test_time_aware.csv', index=False)

        print('Data Split Based on Time Aware, You can now train on (x_train_time_aware, y_train_time_aware) and evaluates on (x_test_time_aware, y_test_time_aware).')