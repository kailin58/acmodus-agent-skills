"""vyton.core.datasets — 数据集接口库(替代29份拷贝)
用法:
    from vyton.core.datasets import ImageFolderDS, PairDS
    ds = ImageFolderDS(files, size=256, labels=optional)  # 单图
    pds = PairDS(pairs_a, pairs_b, size=192)              # 配对
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


def load_tensor(path, size, augment=True):
    """图→[-1,1]张量(标准入口,全项目唯一实现)"""
    im = Image.open(path).convert("RGB").resize((size, size))
    x = torch.tensor(np.asarray(im) / 127.5 - 1, dtype=torch.float32).permute(2, 0, 1)
    if augment and torch.rand(1).item() < 0.5:
        x = torch.flip(x, dims=[2])
    return x


class ImageFolderDS(Dataset):
    """单图数据集:files+(可选多标签dict)"""
    def __init__(self, files, size=160, labels=None, augment=True):
        self.files = [f for f in files if os.path.exists(f)]
        self.size, self.labels, self.augment = size, labels, augment

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        x = load_tensor(self.files[i], self.size, self.augment)
        if self.labels is not None:
            return x, self.labels.get(self.files[i], {})
        return x


class PairDS(Dataset):
    """配对数据集(a=输入,b=目标,同步增强)"""
    def __init__(self, files_a, files_b, size=192):
        n_a, n_b = len(files_a), len(files_b)
        if n_a != n_b:
            print(f"[PairDS] 警告: 输入长度不等 a={n_a} b={n_b},截断到{min(n_a,n_b)}")
        self.pairs = [(a, b) for a, b in zip(files_a, files_b)
                      if os.path.exists(a) and os.path.exists(b)]
        self.size = size

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        a, b = self.pairs[i % len(self.pairs)]
        flip = torch.rand(1).item() < 0.5
        out = []
        for p in (a, b):
            x = load_tensor(p, self.size, augment=False)
            if flip:
                x = torch.flip(x, dims=[2])
            out.append(x)
        return out
