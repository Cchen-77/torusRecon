import torch
import numpy
import h5py
from torch import nn 
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch.utils.data import Dataset,DataLoader
from pathlib import Path
from Datasets import *
import time
from tqdm import tqdm,trange

SEED = 7777777
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED)

class ToyTorusNet(nn.Module):
    def __init__(self,point_count):
        super().__init__()
        hidden_size0 = 64
        hidden_size1 = 128
        hidden_size2 = 1024
        self.fc0 = nn.Linear(3, hidden_size0,dtype=torch.float64)
        self.fc1 = nn.Linear(hidden_size0, hidden_size1,dtype=torch.float64)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2,dtype=torch.float64)
        
        self.fc3 = nn.Linear(hidden_size2,hidden_size1,dtype=torch.float64)
        self.fc4 = nn.Linear(hidden_size1,hidden_size0,dtype=torch.float64)
        self.fc5 = nn.Linear(hidden_size0,3,dtype=torch.float64)

        self.relu = nn.ReLU()
        self.to(DEVICE)
        
    def forward(self,x:torch.Tensor):
        x = self.fc0(x)
        x = self.relu(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x,_= torch.max(x,dim=1)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.relu(x)
        x = self.fc5(x)
        return x

def SaveModel(model:ToyTorusNet,path:Path):
    torch.save(model.state_dict(),path)
def LoadModel(path:Path):
    model = ToyTorusNet()
    model.load_state_dict(torch.load(path))
    return model
def y2params_0(y):
    theta =  y[0] - torch.pi*torch.floor(y[0]/torch.pi)
    phi = y[1] - 2*torch.pi*torch.floor(y[1]/(2*torch.pi))
    normal = torch.tensor([torch.sin(theta)*torch.cos(phi),torch.sin(theta)*torch.sin(phi),torch.cos(theta)])
    point = normal*y[2]
    return normal,point
def loss_0(X,y):
    tot_loss = torch.tensor(0,dtype=torch.float64)
    batch_size = len(X)
    point_size = len(X[0])
    for i in range(batch_size):
        y_pred = y[i]
        normal,point = y2params_0(y_pred)
        normalized_d = F.normalize(X[i]-point, p=2, dim=1)
        expanded_normal = normal.unsqueeze(0).expand(point_size, -1)
        elementwise_product = torch.mul(normalized_d, expanded_normal)
        tot_loss += torch.abs(torch.sum(elementwise_product))/point_size
    return tot_loss/batch_size
def acc_0(y_pred,y):
    batch_size = len(y)
    acc = 0
    for i in range(batch_size):
        normal_pred,point_pred = y2params_0(y_pred[i]) 
        normal,point = y2params_0(torch.tensor([y[i][7],y[i][8],0]))
        acc += torch.norm(normal - normal_pred,p=2).item()
    return acc/batch_size

def TrainModel(model:ToyTorusNet,path:Path,epochs=1000):
    dataset = TPCDataset(path)
    train_ratio = 0.8
    train_size = int(train_ratio*len(dataset))
    test_size = len(dataset) - train_size
    train_dataset,test_dataset = torch.utils.data.random_split(dataset,[train_size,test_size])

    train_dataloader = DataLoader(train_dataset,8,shuffle=True)
    test_dataloader = DataLoader(test_dataset,8,shuffle=False)
    model.to(DEVICE)

    loss_fn = loss_0
    optimizer = torch.optim.SGD(model.parameters(),lr = 0.005)
    for epoch in range(epochs):
        print(f"Epoch:{epoch}:")
        model.train()   
        train_loss = 0
        for X,y in tqdm(train_dataloader,desc="Train"):
            X = X.to(DEVICE)
            y = y.to(DEVICE)
            optimizer.zero_grad()
            y_pred = model(X)
            loss = loss_fn(X,y_pred)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
        train_loss /= len(train_dataloader)
        print(f"TrainLoss:{train_loss:.3f}")

        model.eval()
        with torch.inference_mode():
            test_loss = 0
            for X,y in tqdm(test_dataloader,desc="Test"):
                X = X.to(DEVICE)
                y = y.to(DEVICE)
                y_pred = model(X)
                test_loss += loss_fn(X,y_pred).item()/len(test_dataloader)
            test_loss /= len(train_dataloader)
        print(f"TestLoss:{train_loss:.3f}")
            
    return model

        


    



        
