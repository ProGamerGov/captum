from typing import List

import torch


class ImageTestDataset(torch.utils.data.Dataset):
    def __init__(self, tensors: List[torch.Tensor]) -> None:
        assert all(t.size(0) == 1 for t in tensors if t.dim() == 4)
        t_squeeze = lambda x: x.squeeze(0) if x.dim() == 4 else x
        tensors = [t_squeeze(t) for t in tensors]
        self.tensors = tensors

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.tensors[idx]

    def __len__(self) -> int:
        return len(self.tensors)
