import torch
from torch.nn import functional as F
class Seq2imgDataset(torch.utils.data.Dataset):
    def __init__(self,path):
        with open(path,'r') as f:
            self.lines = f.readlines()

    def __getitem__(self,idx):
        line = self.lines[idx]
        values = line.split('\t')
        values = values[1:] # remove the ID
        attr1, attr2, y = values[:14719], values[14719:-1], values[-1]
        
        # 3.1 < '3.1'
        attr1 = list(map(float,attr1))
        attr2 = list(map(float,attr2))
        y     = float(y)
        # to tensor
        attr1 = torch.tensor(attr1+[ 1. for _ in range(43)]) # 14719 + 43 = 14762 = 122x121
        attr2 = torch.tensor(attr2+[ 0. for _ in range(43)])
        y     = torch.tensor(y)
        y     = y.unsqueeze(-1)
        # reshape to 2D
        attr1 = attr1.reshape(122,121)
        attr2 = attr2.reshape(122,121)
        
        # 3 channel [att1,attr2,attr2]
        x = torch.stack([attr1,attr2,attr2])
        
        return x,y
    
    def __len__(self):
        return len(self.lines)
   