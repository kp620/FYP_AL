import Data_Processing_sub
import os

ids17_url = "https://intrusion-detection.distrinet-research.be/WTMC2021/Dataset/dataset.zip"
download_path = "dataset.zip"
extract_to_directory = "dataset"



def fetch_process_data():
    if os.path.exists("dataset"):
        print("Dataset already exists!")
    else:
        get_data_tool = Data_Processing_sub.GetData(label_mode = "binary", url = ids17_url, download_path = download_path, extract_to_directory = extract_to_directory)
        get_data_tool.download_and_unzip_dataset()

    if os.path.exists("dataset/clean_data.csv"):
        print("Cleaned data already exists!")
    else:
        get_data_tool = Data_Processing_sub.GetData(label_mode = "binary", url = ids17_url, download_path = download_path, extract_to_directory = extract_to_directory)
        get_data_tool.prepare_data()
        
    if os.path.exists("dataset/x_train_time_aware.csv"):
        print("Time aware split already done!")
    else:
        split_data_tool = Data_Processing_sub.SplitData(label_mode = "binary", url = ids17_url, download_path = download_path, extract_to_directory = extract_to_directory)
        split_data_tool.time_aware_split()

fetch_process_data()