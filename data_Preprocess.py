import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, Subset
import indexed_Dataset 
from scipy.spatial.distance import cdist
import subprocess
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# Load training data
def load_data(class_type):
    print("Loading data...")
    data_dic_path = "/vol/bitbucket/kp620/FYP/dataset"
    if(class_type == "binary"):
        x_data = pd.read_csv(f'{data_dic_path}/x_data_iid_binary.csv').astype(float)
        y_data = pd.read_csv(f'{data_dic_path}/y_data_iid_binary.csv').astype(float)
    elif(class_type == "multi"):
        x_data = pd.read_csv(f'{data_dic_path}/x_data_iid_multiclass.csv').astype(float)
        y_data = pd.read_csv(f'{data_dic_path}/y_data_iid_multiclass.csv').astype(float)
    print('Full Data Loaded!')
    print('Fulll Data Length: ', len(x_data))
    # x_data = torch.from_numpy(x_data.values).unsqueeze(1)
    x_data = torch.from_numpy(x_data.values)
    y_data = torch.from_numpy(y_data.values)
    unique_labels = y_data.unique().tolist()
    print("Unique labels: ", len(unique_labels))
    full_dataset = TensorDataset(x_data, y_data)
    return full_dataset

def self_PCA(full_dataset, d = 3):
    x_data = full_dataset.tensors[0].numpy()
    x_data_mean = np.mean(x_data, axis=0)
    x_data_centered = x_data - x_data_mean
    cov = np.cov(x_data_centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvecs = eigvecs[:, np.argsort(eigvals)[::-1]]
    eigvecs_d = eigvecs[:, :d]
    x_data_reduced = np.dot(x_data_centered, eigvecs_d)
    full_dataset_pca = TensorDataset(torch.from_numpy(x_data_reduced), full_dataset.tensors[1])
    return full_dataset_pca
    

def Kmeansplusplus(full_dataset_pca, K=13):
    # Kmeans++ initialization
    x_data = full_dataset_pca.tensors[0].numpy()
    N = len(x_data)
    # Randomly choose the first centroid
    centroids = [x_data[np.random.choice(N)]]
    # Choose the next K-1 centroids
    for _ in range(K-1):
        # Compute the distance of each point to the nearest centroid
        distances = np.array([min([np.linalg.norm(x-c) for c in centroids]) for x in x_data])
        # Choose the next centroid with probability proportional to the distance
        probabilities = distances**2
        probabilities /= probabilities.sum()
        centroids.append(x_data[np.random.choice(N, p=probabilities)])
    return centroids

def mini_K(centroids, full_dataset_pca, max_iters= 10, rs_rate=0.05, K=13):
    x_data = full_dataset_pca.tensors[0].numpy()
    v = np.zeros(K)  # Count
    n = 1 / (v + 1)  # Learning rate (n)
    C = centroids  # Centers
    t = 0
    for t in range(max_iters):
        # Select b instances randomly
        indices = np.random.choice(x_data.shape[0], int(rs_rate*x_data.shape[0]), replace=False)
        M = x_data[indices]

        # Cache the closest center to each point in the mini-batch
        d = {i: np.argmin(np.linalg.norm(x-C, axis=1)) for i, x in enumerate(M)}

        # Update per-center counts and learning rates
        for i, x in enumerate(M):
            c_index = d[i]
            v[c_index] += 1
            n = 1 / v[c_index]  # Update learning rate for this center

            # Take gradient step
            C[c_index] = (1 - n) * C[c_index] + n * x
    return C

def cluster_sample(full_dataset, full_dataset_pca, C, batch_size, rs_rate = 0.05):
    x_data = full_dataset_pca.tensors[0].numpy()
    distances = cdist(x_data, C, 'euclidean')  # Compute all distances from data points to centroids
    cluster_labels = np.argmin(distances, axis=1)  # Assign to the nearest centroid

    full_length = len(full_dataset_pca)
    indices_list = []

    for cluster_id in range(len(C)):
    # Indices for all points in this cluster
        indices_in_cluster = np.where(cluster_labels == cluster_id)[0]
        
        # Define your sample size
        sample_size = int(len(indices_in_cluster) * rs_rate)

        # Sample indices
        sampled_indices = np.random.choice(indices_in_cluster, size=sample_size, replace=False)
        indices_list.append(sampled_indices)


    # ----------------- visualization -----------------

    # pca = PCA(n_components=2)
    # x_reduced = pca.fit_transform(x_data)

    # plt.figure(figsize=(10, 8))
    # scatter = plt.scatter(x_reduced[:, 0], x_reduced[:, 1], c=cluster_labels, alpha=0.5, cmap='viridis')
    # plt.colorbar(scatter)
    # plt.title('Cluster Visualization')
    # plt.xlabel('Principal Component 1')
    # plt.ylabel('Principal Component 2')
    # plt.show()
    # plt.savefig('/vol/bitbucket/kp620/FYP/cluster.png', format='png', dpi=300)  # Adjust format and dpi as needed
    # plt.close()

    # ----------------- visualization -----------------
    x_data = full_dataset.tensors[0].numpy()
    x_data = torch.from_numpy(x_data).unsqueeze(1)
    command = "echo 'x_data shape: " + str(x_data.shape) + "'"
    subprocess.call(command, shell=True)
    full_dataset = TensorDataset(x_data, full_dataset.tensors[1])
    # not_selected_indice = np.setdiff1d(np.arange(full_length), np.concatenate(indices_list))
    
    indices_list = [np.array(item).flatten() for item in indices_list]
    selected_indices = np.concatenate(indices_list)
    not_selected_indice = np.setdiff1d(np.arange(full_length), selected_indices)

    np.save('/vol/bitbucket/kp620/FYP/dataset/not_selected_indice.npy', not_selected_indice)
    np.save('/vol/bitbucket/kp620/FYP/dataset/selected_indice.npy', selected_indices)
    


def main(batch_size, rs_rate, class_type):
    full_dataset = load_data(class_type)
    command = "echo 'loading finished'"
    subprocess.call(command, shell=True)

    full_dataset_pca = self_PCA(full_dataset, d = 10)
    command = "echo 'PCA finished'"
    subprocess.call(command, shell=True)

    centroids = Kmeansplusplus(full_dataset_pca)
    command = "echo 'Kmeans++ finished'"
    subprocess.call(command, shell=True)

    C = mini_K(centroids, full_dataset_pca)
    command = "echo 'mini_K finished'"
    subprocess.call(command, shell=True)

    cluster_sample(full_dataset, full_dataset_pca, C, batch_size=batch_size, rs_rate = rs_rate)


main(1024, 0.001, "multi")