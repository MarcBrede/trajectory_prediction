from torch.utils.data import Dataset, DataLoader
import torch
import random

class GNN_Dataset_Trajectory(Dataset):
    # data is a dict with key: (number_of_entities, number_of_vehicles) and value: (X [length, 4], batch [length/number_of_entities, 4])
    # self.data is a list of datapoints (X, batch)
    def __init__(self, data):
        super().__init__()
        self.data = []
        for key, value in data.items():
            number_entities, number_vehicles = key
            X, batch = value
            assert X.shape[0] == batch.shape[0] * number_entities, "This does not match!"

            N = batch.shape[0]
            X = X.reshape(N, number_entities, 5)
            batch = batch.type(torch.int) 

            self.data.extend(list(zip(X, batch)))
        
        random.shuffle(self.data)
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

def collect_fn(batch):
    X, n = zip(*batch)
    X = torch.cat(X)
    n = torch.stack(n)
    return (X, n)

class GNN_Dataloader(DataLoader):
    def __init__(self, dataset, batch_size, shuffle=False, drop_last=False):
        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            collate_fn=collect_fn
        )       

