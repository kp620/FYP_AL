"""
IndexedDataset class
Self-defined class to return the index of the data in the dataset
"""

class IndexedDataset():
    # Initialize the dataset
    def __init__(self, dataset):
        self.dataset = dataset
    
    # Get the data and target and index of the dataset
    def __getitem__(self, index):
        data, target = self.dataset[index]
        return data, target, index

    # Get the length of the dataset
    def __len__(self):
        return len(self.dataset)
