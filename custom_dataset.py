from torch.utils.data import Dataset
import torch

class StyleIdxDataset(Dataset):
  def __init__(self, num_of_samples, n_style):
    self.data = torch.arange(num_of_samples)
    self.n_style = n_style
    
  def __len__(self):
    return len(self.data)
  
  def __getitem__(self, idx):
    return self.data[idx]%self.n_style