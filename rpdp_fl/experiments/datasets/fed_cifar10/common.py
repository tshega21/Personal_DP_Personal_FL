import torch
from .dataset import FedCifar10, Cifar10Raw

#Root Mean Square Propagation
Optimizer = torch.optim.RMSprop
FedClass = FedCifar10
RawClass = Cifar10Raw


