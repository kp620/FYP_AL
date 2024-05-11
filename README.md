# FYP_AL

## Files explanation 
- 'approxx_Optimizer.py'
Appximate the hessian signal

- 'data_Fetch.py'
Download the CICIDS17 data

- 'data_Preprocess.py'
Apply PCA and Kmenas++ algorithm to the full dataset. Use as initial clustering.

- 'facility_Update.py'
Apply the facility location function, return the selected index and corresponding weights.

- 'indexed_Dataset.py'
Self-implemented dataset class.

- 'model_Trainer.py'
Encapsulate the operations of the model.

- 'restnet_1d.py'
Define the model. 

- 'uncertainty_similarity.py'
Calculate and combine the uncertainty and similarity factors. 

## Usage
1. Get the CICIDS17 dataset
2. Preprocess the data: 
e.g. 'python data_Preprocess.py --rs_rate=0.001 --d=10 --K=15'
3. Run the trainer
e.g. 'python new_trainer_trusted_zone.py --operation_type='iid' --class_type='multi' --budget=0.009'


## Hyperparameters
- 