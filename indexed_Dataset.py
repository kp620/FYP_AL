"""
IndexedDataset class
Self-defined class to return the index of the data in the dataset
"""

class IndexedDataset():
    def __init__(self, dataset):
        self.dataset = dataset
    
    def __getitem__(self, index):
        data, target = self.dataset[index]
        return data, target, index

    def __len__(self):
        return len(self.dataset)
