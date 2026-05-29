import torch
from .dataset import FedFashionMnist, FashionMnistRaw
Optimizer = torch.optim.Adam
FedClass = FedFashionMnist
RawClass = FashionMnistRaw