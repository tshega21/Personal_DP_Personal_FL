import argparse
import datetime
import importlib
import numpy as np
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
import warnings
warnings.simplefilter("ignore")

from configs.config_utils import read_config, get_config_file_path
from myopacus.strategies import Per_FedAvg

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default='heart_disease')
parser.add_argument("--gpuid", type=int, default=7,
                    help="Index of the GPU device.")
parser.add_argument(
    "--data_type", type=str, default="iid_10",
    choices=[
        "niid_10_5",
        "niid_10_2",
        "niid_dir_1",
        "niid_dir_5",
        "iid_10",
    ],
    help="Dataset partition type"
)
parser.add_argument("--reg_param", type=float, default=0.1, 
                    help="regularization parameter")
parser.add_argument("--meta_learning_rate", type=float, default=0.1, 
                    help="meta learning rate")
parser.add_argument("--num_personal_steps", type=int, default=5, 
                    help="random seed")
parser.add_argument("--seed", type=int, default=42, 
                    help="random seed")
args = parser.parse_args()

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
def set_random_seed(seed_value):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
set_random_seed(args.seed)
device = torch.device(f"cuda:{args.gpuid}" if torch.cuda.is_available() else "cpu")
module_name = f"datasets.fed_{args.dataset}"
try:
    dataset_modules = importlib.import_module(module_name)
    FedClass = dataset_modules.FedClass
    RawClass = dataset_modules.RawClass
    BaselineModel = dataset_modules.BaselineModel
    BaselineLoss = dataset_modules.BaselineLoss
    Optimizer = dataset_modules.Optimizer
    metric = dataset_modules.metric
    
except ModuleNotFoundError as e:
    print(f'{module_name} import failed: {e}')

project_abspath = os.path.abspath(os.path.join(os.getcwd(),".."))
dict = read_config(get_config_file_path(dataset_name=f"fed_{args.dataset}", debug=False))
# save_dir

opt_method = "per_fedavg"
# Base save directory from config "results folder"
base_save_dir = os.path.join(project_abspath, dict["save_dir"])

# Subfolder by dataset type (e.g., iid / niid)
save_dir = os.path.join(base_save_dir, args.data_type,opt_method)

# Ensure the folder exists
os.makedirs(save_dir, exist_ok=True)



save_file = os.path.join(save_dir, f"{args.data_type}_results_per_fedavg_vanilla_{args.dataset}.csv")



NUM_CLIENTS = dict["fedavg"]["num_clients"]
NUM_STEPS = dict["fedavg"]["num_steps"]
NUM_ROUNDS = dict["fedavg"]["num_rounds"]
CLIENT_RATE = dict["fedavg"]["client_rate"]
BATCH_SIZE = dict["fedavg"]["batch_size"]
LR = dict["fedavg"]["learning_rate"]

# data_dir
if args.dataset == "heart_disease":
    data_path = os.path.join(project_abspath, dict["dataset_dir"])
else:
    data_path = os.path.join(project_abspath, dict["dataset_dir"][f"{args.data_type}"]) 


rawdata = RawClass(data_path=data_path)
training_dls, test_dls = [], []



for i in range(NUM_CLIENTS):
    g = torch.Generator()
    seed = 1000+i
    g.manual_seed(seed)

    
    train_ds = FedClass(rawdata=rawdata, center=i, train=True)
    #print(len(train_ds))
    test_ds = FedClass(rawdata=rawdata, center=i, train=False)
    
    training_dls.append(DataLoader(train_ds,batch_size=BATCH_SIZE,shuffle=True,generator=g))
    test_dls.append(DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False))

    
# creates Pooled dataset with data from all clients
pooled_test_ds = FedClass(rawdata, pooled=True, train=False)
pooled_test_dl = DataLoader(pooled_test_ds, batch_size=BATCH_SIZE)


#Prepare model and loss

model = BaselineModel.to(device)
criterion = BaselineLoss()

results_all_reps = []
current_args = {

    "training_dataloaders": training_dls,
    "test_dataloaders": test_dls,
    "model": model,
    "loss": criterion,
    "optimizer_class": Optimizer,
    "learning_rate": LR,
    "num_steps": NUM_STEPS,
    "num_rounds": NUM_ROUNDS,
    "client_rate": CLIENT_RATE,
    "device": device,
    "metric": metric,
    "seed": args.seed
}
current_args["reg_param"] = args.reg_param
current_args["num_personal_steps"] = args.num_personal_steps
current_args["pooled_test_dataloader"] = pooled_test_dl
print("learning rate = ", current_args["learning_rate"])

# ======== Start Training ==========
s = Ditto(**current_args, log=False)
cm, perf_global, perf_personal, perf_pooled = s.run()

perf_data = {
    "global": perf_global,
    "personal": perf_personal,
    "pooled": perf_pooled
}

# Save records for global, personal, and personal pooled acurracy 
for key in ["global", "personal", "pooled"]:
    perf = perf_data[key]
    mean_perf = np.mean(perf[-3:])
    
    print(f"All rounds performance of vanilla per fedavg {key}, Perf={perf}")
    print(f"Mean performance of vanilla per fedavg {key}, Perf={mean_perf:.4f}")

    
    record = [{ 
    "type": key,
    "lambda": args.reg_param,
    "num_personal_steps": args.num_personal_steps,
    "perf": str(perf),
    "mean_perf": round(mean_perf, 4),
    "e": "PrivacyFree",
    "d": None,
    "nm": None,
    "norm": None,
    "bs": BATCH_SIZE,
    "seed": args.seed
    }]

    results_df = pd.DataFrame.from_dict(record)
    
    # Append to CSV
    write_header = not os.path.exists(save_file)
    results_df.to_csv(save_file, mode='a', index=False, header = write_header)

    
# ======== End Training ============
