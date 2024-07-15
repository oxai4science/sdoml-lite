import torch
from torch.utils.data import Dataset
import webdataset as wds
import os
from glob import glob


class SDOMLlite(Dataset):
    def __init__(self, files):
        self.dataset = (wds.WebDataset(files)
                        .decode() # Handles npy decoding automatically
        )

# WORK IN PROGRESS