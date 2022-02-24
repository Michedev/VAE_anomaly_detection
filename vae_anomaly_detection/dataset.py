import torch
from torch.utils.data import Dataset, TensorDataset


def rand_dataset(num_rows=60_000, num_columns=100) -> Dataset:
    return TensorDataset(torch.rand(num_rows, num_columns))