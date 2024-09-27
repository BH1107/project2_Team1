import torch
from torch.utils.data import DataLoader, Dataset
import lightning as L

class CostomerDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        X = torch.from_numpy(self.data.iloc[idx].drop('Churn').values).float()
        y = torch.Tensor([self.data.iloc[idx].Churn]).float()
        
        return {
            'X': X,
            'y': y,
        }


class CostomerDataModule(L.LightningDataModule):
    def __init__(self, batch_size: int):
        super().__init__()
        self.batch_size = batch_size

    def prepare(self, train_dataset, valid_dataset, test_dataset):
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset

    def setup(self, stage: str):
        if stage == "fit":
            self.train_data = self.train_dataset
            self.valid_data = self.valid_dataset

        if stage == "test":
            self.test_data = self.test_dataset

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_data,
            batch_size=self.batch_size,
            shuffle=True,  
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.valid_data,
            batch_size=self.batch_size,
            shuffle=False, 
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_data,
            batch_size=self.batch_size,
            shuffle=False,
        )
