import requests
import zipfile
import os
import json
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# with open('config.json', 'r') as config_file:
#     config = json.load(config_file)

ids17_url = "https://intrusion-detection.distrinet-research.be/WTMC2021/Dataset/dataset.zip"
download_path = "dataset.zip"
extract_to_directory = "/vol/bitbucket/kp620/FYP/dataset"

config = {
            "ids17_url": ids17_url,
            "download_path": download_path,
            "extract_to_directory": extract_to_directory
        }

def download_and_unzip_dataset():
    url = config['ids17_url']
    download_path = config['download_path']
    extract_to = config['extract_to_directory']

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



def set_pandas_options():
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

def read_and_clean_data(file_path):
    df = pd.read_csv(file_path)
    df = df.drop_duplicates().dropna()
    return df

def combine_dataframes(dfs):
    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df

def label_data(data):
    original_labels = data['Label'].copy()
    data["Label"] = data["Label"].apply(lambda x: 0 if x == "BENIGN" else 1)
    return data, original_labels

def process_data(df):
    attempted_labels = [s for s in df['Label'].unique() if 'Attempted' in s]
    df.replace(attempted_labels, 'BENIGN', inplace=True)
    cols_to_remove = ['Flow ID', 'Src IP', 'Src Port', 'Dst IP', 'Timestamp']
    df = df.drop(cols_to_remove, axis=1)
    return df


def prepare_data():
    set_pandas_options()
    print('Processing the Data')
    ds_dic = config['extract_to_directory']
    file_paths = [f"{ds_dic}/Monday-WorkingHours.csv", f"{ds_dic}/Tuesday-WorkingHours.csv", f"{ds_dic}/Wednesday-WorkingHours.csv", f"{ds_dic}/Thursday-WorkingHours.csv", f"{ds_dic}/Friday-WorkingHours.csv"]
    dataframes = [read_and_clean_data(file_path) for file_path in file_paths]
    for i, df in enumerate(dataframes, 1):
        print(f'Day {i}: {len(df)} rows')
    combined_df = combine_dataframes(dataframes)
    processed_df = process_data(combined_df)
    nRow, nCol = processed_df.shape
    print(f'Processed data has {nRow} rows and {nCol} columns')
    processed_df, original_labels = label_data(processed_df)
    processed_df.to_csv(f'{ds_dic}/clean_data_binary.csv', index=False)
    original_labels.to_csv(f'{ds_dic}/original_labels.csv', index=False)
    print(f' Data Saved: clean_data.csv and original_labels.csv')

def split_data(split_type):
    """
    Split Methods Can Be Either Time Aware or IID (Independent and identically distributed)

    Time Aware: Train The Initial Model On The First two days (x_train, y_train)
    and Test&Select from The Last Three Days.

    IID: Here the data splited randomly, we have three sets: hold_out, train_set, and select_set.
    The hold_out set must not be used either in training or optimization (e.g. early stopping)
    In this scenario, you train on the train_set (x_train_iid.csv, y_train_iid.csv),
    and  you can select from select_set, and evaluate the performance on hold_out.

    """
    #split_type = config["split_type"]
    ds_dic = config['extract_to_directory']
    data = pd.read_csv(f'{ds_dic}/clean_data_binary.csv')
    original_labels = pd.read_csv(f'{ds_dic}/original_labels.csv')
    data = data.astype(float)
    print('Split and Scale Data')
    if split_type == "iid":
       print('Start IID Split')
       y_data, x_data = data["Label"], data.drop('Label', axis=1)
       x_data.replace([np.inf, -np.inf], np.nan, inplace=True)
       x_data = np.nan_to_num(x_data)
       scaler = MinMaxScaler().fit(x_data)
       x_data = scaler.transform(x_data)
       x_data = pd.DataFrame(x_data)
       y_data = pd.DataFrame(y_data)
       x_data.to_csv(f'{ds_dic}/x_data_iid_binary.csv', index=False)
       y_data.to_csv(f'{ds_dic}/y_data_iid_binary.csv', index=False)
    elif split_type == "time-aware":
       print('Start Time Aware Split')
       time_aware = config["time-aware"]
       train_data = data[:time_aware]
       train_original_labels = original_labels[:time_aware]
       y_train, x_train = train_data["Label"], train_data.drop('Label', axis=1)
       test_data = data[time_aware:]
       test_original_labels = original_labels[time_aware:]
       y_test, x_test = test_data["Label"], test_data.drop('Label', axis=1)
       x_test.replace([np.inf, -np.inf], np.nan, inplace=True)
       x_test = np.nan_to_num(x_test)
       x_train.replace([np.inf, -np.inf], np.nan, inplace=True)
       x_train = np.nan_to_num(x_train)

       scaler = MinMaxScaler().fit(x_train)
       x_train, x_test = scaler.transform(x_train), scaler.transform(x_test)
       x_train_df = pd.DataFrame(x_train)
       x_test_df = pd.DataFrame(x_test)
       x_train_df.to_csv(f'{ds_dic}/x_train_time_aware.csv', index=False)
       x_test_df.to_csv(f'{ds_dic}/x_test_time_aware.csv', index=False)
       y_train.to_csv(f'{ds_dic}/y_train_time_aware.csv', index=False)
       y_test.to_csv(f'{ds_dic}/y_test_time_aware.csv', index=False) 
       train_original_labels.to_csv(f'{ds_dic}/y_train_time_aware_original.csv', index=False)
       test_original_labels.to_csv(f'{ds_dic}/y_test_time_aware_original.csv', index=False)

       print('Data Split Based on Time Aware, You can now train on (x_train_time_aware, y_train_time_aware) and evaluates on (x_test_time_aware, y_test_time_aware).')
    else:
        print(f"Print Wrong Choice: {split_type}, Available Choices: iid and time-aware")
    

def main(download, prepare, split, all_option, split_type):
    if all_option:
        if split_type is None:
           print('Please specify the split type with --split_type, Choices: "iid" or "time-aware"')
           return
        download_and_unzip_dataset()
        prepare_data()
        split_data(split_type)
    else:
        if download:
            download_and_unzip_dataset()
        if prepare:
            prepare_data()
        if split:
            if split_type is not None:
                split_data(split_type)
            else:
                print('Please specify the split type with --split_type, Choices: "iid" or "time-aware"')
                return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run data processing steps.")
    parser.add_argument('--download', action='store_true', help='Download and unzip dataset')
    parser.add_argument('--prepare', action='store_true', help='Prepare data')
    parser.add_argument('--split', action='store_true', help='Split data')
    parser.add_argument('--all', action='store_true', help='Run all steps')
    parser.add_argument('--split_type', choices=['iid', 'time-aware'], help='Specify split type: "iid" or "time-aware"')

    args = parser.parse_args()

    if args.split and not args.split_type:
        parser.error('--split requires --split_type to be specified.')
    else:
        main(args.download, args.prepare, args.split, args.all, args.split_type)
