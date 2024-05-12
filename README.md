# FYP_AL

## Files explanation 
- `approxx_Optimizer.py`
Appximate the hessian signal

- `data_Fetch.py`
Download the CICIDS17 data

- `data_Preprocess.py`
Apply PCA and Kmenas++ algorithm to the full dataset. Use as initial clustering.

- `facility_Update.py`
Apply the facility location function, return the selected index and corresponding weights.

- `indexed_Dataset.py`
Self-implemented dataset class.

- `model_Trainer.py`
Encapsulate the operations of the model.

- `restnet_1d.py`
Define the model. 

- `uncertainty_similarity.py`
Calculate and combine the uncertainty and similarity factors. 

- `new_trainer_trusted_zone.py`
Main trainer using trusted zone method.

- `not_selected_indice.npy`, `selected_indice.npy`
The indices of the sample selected for initial training

- `x_data_iid_multiclass.csv`, `y_data_iid_multiclass.csv`
The full multiclass CICIDS17 data file

## Usage
1. Get the CICIDS17 dataset
e.g. `python data_Fetch.py --all --split_type='iid'`
2. Preprocess the data: 
e.g. `python data_Preprocess.py --rs_rate=0.001 --d=10 --K=15`
3. Run the trainer
e.g. `python new_trainer_trusted_zone.py --operation_type='iid' --class_type='multi' --budget=0.009`

### Note
The initial training set is 0.1%. So **budget = real_budget - 0.001**
e.g. if want to select 1% data, then budget = 0.01 - 0.001 = 0.009


## Hyperparameters
- `batch_size`: batch size
- `lr`: learning rate
- `budget`: the maximum number of samples to be selected, determined by 'budget_ratio'
- `budget_ratio`: the maximum rate of selection. 
- `gf, ggf, ggf_moment`: hessian related parameters
- `delta`: trusted zone
- `stop`: signs of reaching budget
- `alpha, alpha_max`: quantify the weights between similarity and uncertainty
- `sigma`: scaling factors in Guassian, determines how the distance between two points is translated into a similarity measure.
- `eigenv`: stores the eigenvectors calculated from PCA