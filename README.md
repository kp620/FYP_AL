
# Enhancing Network Intrusion Detection Systems with Active Learning: A Semi-Supervised Coreset Selection Approach

## Abstract

The rapid evolution of cyber threats necessitates advanced methodologies for Network Intrusion Detection Systems (NIDS). While deep learning offers promising solutions, the scarcity and high cost of labeled data pose significant challenges. This thesis investigates the application of active learning and coreset selection to enhance the efficiency and effectiveness of NIDS, particularly in scenarios with limited labeled data.


We propose a semi-supervised coreset selection approach that strategically selects a representative subset of the dataset, preserving its statistical properties and informativeness. Experiments are conducted under both Independent and Identically Distributed (IID) and Time-Aware (TA) settings to access the robustness of the approach. Our methodology demonstrates effectiveness when applied to the CICIDS17 dataset, and shows generalization ability when tested on a malware dataset derived from APIGraph. This thesis highlights the potential of active learning frameworks in optimizing data utilization for machine learning techniques in cybersecurity, paving the way for more efficient and scalable NIDS solutions.



## Documentation

### Auxiliary Files
- `approx_Optimizer.py`
CITE: Yao Z, Gholami A, Shen S, Mustafa M, Keutzer K, Mahoney M. Adahessian: An adaptive second order optimizer for machine learning. Inproceedings of the AAAI conference on artificial intelligence 2021 May 18 (Vol. 35, No. 12, pp. 10665-10673).

- `facility_Update.py`
Selects instances that maximize the facility location function, which is a submodular function that models the diversity of a subset of instances.

- `indexed_Dataset.py`
Self-defined class to return the index of the data in the dataset

- `test_Cuda.py`
Check the device and data type.

- `uncertainty_similarity.py`
Calculate uncertainties and similarities for each unlabelled sample. Output continuous states for each unlabelled sample.


### Experiment Files

- `IDS_IID/new_trainer_trusted_zone.py`
Perform IID experiment over CICIDS17 dataset. 

- `IDS_TA/new_trainer_trusted_zone_ta.py`
Perform TA experiment over CICIDS17 dataset. 

- `Android_workspace/new_trainer_trusted_zone_ta.py`
Perform TA experiment over Malware dataset. 
## Running Tests

### Fetch CICIDS17 Data
`python data_Fetch.py --all --split_type='iid'`

### IDS_IID Test
In order to perform the IDS experiment under IID setting, please follow the following steps.

`python data_Preprocess.py --rs_rate=rs_rate, --d=d, --K=K`

This will first load the IDS dataset, then performs the K-means++ algorithm to form K clusters. From each cluster, a portion samples are chosen to form the initial training set. The selected and non_selected indexes will be stored in local files for future usage. 

rs_rate determines the selection rate of samples to form the initial set. 

d determins the reduced dimension of PCA. 

K determines the number of clusters. 

`python IDS_IID/new_trainer_trusted_zone.py --budget=budget`

This example will perform the IDS IID experiment with the selected budget. 


### IDS_TA Test
In order to perform the IDS experiment under TA setting, please follow the following steps.

`python IDS_TA/new_trainer_trusted_zone_ta.py --budget=budget`

This example will perform the IDS IID experiment with the selected budget. 


### Malware_TA Test
In order to perform the Malware experiment under TA setting, please follow the following steps.

`python Android_workspace/new_trainer_trusted_zone_ta.py --budget=budget`

This example will perform the Malware TA experiment with the selected budget. 
