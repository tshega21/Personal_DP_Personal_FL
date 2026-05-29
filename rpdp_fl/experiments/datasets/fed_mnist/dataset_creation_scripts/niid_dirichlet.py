import numpy as np
import os
import random
from torchvision.datasets import MNIST

np_rng = np.random.default_rng(42)

NUM_CLIENTS = 10
NUM_LABELS = 10
BATCH_SIZE = 200
ALPHA = 0.5  # Dirichlet concentration parameter

dataset_abspath = os.path.abspath(os.path.join(os.getcwd(),"../.."))
data_path = os.path.join(dataset_abspath, "mnist")
if not os.path.exists(data_path):
    os.mkdir(data_path)

train_data = MNIST(data_path, train=True, download=True)
data, target = train_data.data, train_data.targets

save_num = ALPHA*10

save_path = os.path.join(
    os.path.abspath(os.path.join(os.getcwd(),"..")),
    f"dirichlet_{NUM_CLIENTS}_alpha{save_num}"
)
os.makedirs(save_path, exist_ok=True)

# Convert to numpy
data = data.numpy()
target = target.numpy()


# Dirichlet split
min_size = 0
while min_size < BATCH_SIZE:
    client_indices = [[] for _ in range(NUM_CLIENTS)]
    for k in range(NUM_LABELS):
        idx_k = np.where(target == k)[0]
        np_rng.shuffle(idx_k)

        proportions = np_rng.dirichlet(np.repeat(ALPHA, NUM_CLIENTS))
        proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
        split_indices = np.split(idx_k, proportions)

        for cid in range(NUM_CLIENTS):
            client_indices[cid].extend(split_indices[cid])
    min_size = min([len(idx_j) for idx_j in client_indices])


# Save per client
for cid in range(NUM_CLIENTS):
    cname = f'client{cid}'
    indices = client_indices[cid]

    client_X = data[indices]
    client_y = target[indices]

    combined = list(zip(client_X, client_y))
    random.shuffle(combined)

    np.save(os.path.join(save_path, f"{cname}.npy"),
            np.array(combined, dtype=object))

    print(f"{cname}: {np.unique(client_y, return_counts=True)}, size={len(indices)}")