from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class TokenDataset(Dataset):
    def __init__(self, data_dir, split, block_size):
        self.path = Path(data_dir) / f"{split}.bin"
        self.data = np.memmap(self.path, dtype=np.uint16, mode="r")
        self.block_size = block_size
        self.num_blocks = (len(self.data) - 1) // block_size
        if self.num_blocks <= 0:
            raise ValueError(
                f"{self.path} needs at least {block_size + 1} tokens"
            )

    def __len__(self):
        return self.num_blocks

    def __getitem__(self, index):
        start = index * self.block_size
        tokens = np.array(
            self.data[start : start + self.block_size + 1], dtype=np.int64
        )
        x = torch.from_numpy(tokens[:-1])
        y = torch.from_numpy(tokens[1:])
        return x, y
