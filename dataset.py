import torch
from torch.utils.data import Dataset

class CTRDataset(Dataset):
    def __init__(self, num_samples = 5000):

        # fake feature
        # [user_id, item_id, category_id]   - three features of eacn sampe
        # [0, 100)
        self.x = torch.randint(0, 100, (num_samples, 3))



        # fake click label
        # means which click feature is choosed
        self.y = (
            (self.x[:, 0] + self.x[:, 1]) % 2
        ).float()

    # must let the dataloader know the length of the dataset
    def __len__(self):
        # length of the dataset
        return len(self.y)

    # must let the dataloader know how to get a sample by index
    def __getitem__(self, idx):
        # implement the getitem function of Dataset
        return (
            self.x[idx],
            self.y[idx]
        )
