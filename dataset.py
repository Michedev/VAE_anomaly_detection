import torch
from torch.utils.data import Dataset, TensorDataset


def load_dataset() -> Dataset:
    ## Overwrite this to load your dataset
    return TensorDataset(torch.rand(60_000, 100))