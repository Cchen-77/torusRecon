import torch
import numpy
import h5py
from pathlib import Path
from torch.utils.data import Dataset
class TPCDataset(Dataset):
    def __init__(self,path:Path):
        super().__init__()
        with h5py.File(path) as hf:
            self.X = torch.tensor(hf["X"][:]) 
            self.y = torch.tensor(hf["y"][:])
    def __len__(self):
        assert(len(self.X) == len(self.y))
        return len(self.X)
    def __getitem__(self, index):
        return self.X[index],self.y[index]
