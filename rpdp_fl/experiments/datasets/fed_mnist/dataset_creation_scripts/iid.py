import numpy as np
import os
from torchvision.datasets import MNIST

NUM_LABELS = 10
NUM_CLIENTS = 10

dataset_abspath = os.path.abspath(os.path.join(os.getcwd(),"../.."))
data_path = os.path.join(dataset_abspath, "mnist")
if not os.path.exists(data_path):
    os.mkdir(data_path)


# Tensors of 60000 images each 28x28
train_data = MNIST(data_path, train=True, download=True)
#splits into data and target where target is the label
data, target = train_data.data, train_data.targets

# Create federated datasets
save_path = os.path.join(os.path.abspath(os.path.join(os.getcwd(),"..")), f"iid_{NUM_CLIENTS}")
if not (os.path.exists(save_path)):
    os.mkdir(save_path)

num_examples = len(train_data)
num_examples_per_client = num_examples // NUM_CLIENTS
perm = np.random.permutation(num_examples)
for cid in range(NUM_CLIENTS):

    #Splits dataset by client such that each gets disjoint set of shuffled dataset
    indices = np.array(perm[cid * num_examples_per_client : (cid+1) * num_examples_per_client])
    #gets training data for each client
    client_X = data.numpy()[perm[cid * num_examples_per_client : (cid+1) * num_examples_per_client]]
    #gets labels for each client
    client_y = target.numpy()[perm[cid * num_examples_per_client : (cid+1) * num_examples_per_client]]
    combined = list(zip(client_X, client_y)) 

    cname = 'client{:d}'.format(cid) 
    np.save(os.path.join(save_path, f"{cname}.npy"), np.array(combined, dtype=object))