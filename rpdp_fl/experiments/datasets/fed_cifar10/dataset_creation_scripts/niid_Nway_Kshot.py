import numpy as np
import os
import random

from torchvision.datasets import CIFAR10

#WAY = 5
WAY = 2
#SHOT = 1200
SHOT = 1000
NUM_LABELS = 10
NUM_CLIENTS = 10

dataset_abspath = os.path.abspath(os.path.join(os.getcwd(),"../.."))
data_path = os.path.join(dataset_abspath, "cifar10")
if not os.path.exists(data_path):
    os.mkdir(data_path)
train_data = CIFAR10(data_path, train=True, download=True)
data, target = train_data.data, np.array(train_data.targets)

# Create federated datasets
save_path = os.path.join(os.path.abspath(os.path.join(os.getcwd(),"..")), f"niid_{NUM_CLIENTS}_{WAY}way_{SHOT}shot")
if not (os.path.exists(save_path)):
    os.mkdir(save_path)
np_rng = np.random.default_rng(42)


NUM_SHARDS = NUM_CLIENTS * WAY 
NUM_SHARDS_PER_LABEL = len(data) // NUM_LABELS // SHOT

data_tr, data_ts = [], []
target_tr, target_ts = [], []

for i in range(NUM_LABELS):
    idx = target==i
    data_tr.append(list(data[idx]))
    target_tr.append(list(target[idx]))

while(True):
    ###### SPLIT DATA #######
    X = [[] for _ in range(NUM_CLIENTS)]
    y = [[] for _ in range(NUM_CLIENTS)]

    idx = np.zeros(NUM_LABELS, dtype=np.int64)
    shards_index = list(range(NUM_SHARDS))
    user_labels_count = {i: [] for i in range(NUM_CLIENTS)}
    FINDED = True
    for user in range(NUM_CLIENTS):
        chosen_labels = np_rng.choice(NUM_LABELS, WAY, replace=False)

        for lbl in chosen_labels:
            start = idx[lbl]
            end = start + SHOT

            X[user].extend(data_tr[lbl][start:end])
            y[user].extend([lbl] * SHOT)

            idx[lbl] += SHOT
    
    if FINDED:
        break
    
for cid in range(NUM_CLIENTS):
    cname = 'client{:d}'.format(cid) 
    client_X = X[cid]
    client_y = y[cid]

    combined = list(zip(client_X, client_y)) 
    random.shuffle(combined)
    print(cname, len(combined), np.unique(client_y))
    np.save(os.path.join(save_path, f"{cname}.npy"), combined)
    