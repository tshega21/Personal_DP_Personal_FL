import argparse
import copy
import datetime
import importlib
import numpy as np
import os
import pandas as pd
from collections import Counter

import warnings # ignore warnings for clarity
warnings.simplefilter("ignore")

import torch
from torch.utils.data import DataLoader

from configs.config_utils import read_config, get_config_file_path
from myopacus import PrivacyEngine
from myopacus.strategies import FedAvg
from myopacus.strategies import Ditto

from myopacus.accountants.rpdp_utils import GENERATE_EPSILONS_FUNC

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
        "niid_20_5",
        "niid_20_2",
        "niid_20_dir_1",
        "niid_20_dir_5",
        "iid_20"
    ],
    help="Dataset partition type"
)
parser.add_argument("--epsilon", type=float, default=5, 
                    help="epsilon")
parser.add_argument("--reg_param", type=float, default=0.1, 
                    help="regularization parameter")
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

#Configuration file reading
dict = read_config(get_config_file_path(dataset_name=f"fed_{args.dataset}", debug=False))
# save_dir
opt_method = "ditto"

# Base save directory from config "results folder"
base_save_dir = os.path.join(project_abspath, dict["save_dir"])

# Subfolder by dataset type (e.g., iid / niid)
save_dir = os.path.join(base_save_dir, args.data_type,opt_method)

# Ensure the folder exists
os.makedirs(save_dir, exist_ok=True)
save_file = os.path.join(save_dir, f"{args.data_type}_results_ditto_rpdp_data_dep_{args.dataset}.csv")

NUM_CLIENTS = dict["fedavg"]["num_clients"]
NUM_STEPS = dict["fedavg"]["num_steps"]
NUM_ROUNDS = dict["fedavg"]["num_rounds"]
CLIENT_RATE = dict["fedavg"]["client_rate"]
BATCH_SIZE = dict["fedavg"]["batch_size"]
LR = dict["fedavg"]["learning_rate"]

LR_DP = dict["dpfedavg"]["learning_rate"]
MAX_GRAD_NORM = dict["dpfedavg"]["max_grad_norm"]
TARGET_DELTA = dict["dpfedavg"]["target_delta"]
MAX_PHYSICAL_BATCH_SIZE = dict["dpfedavg"]["max_physical_batch_size"]

""" Prepare local datasets """
# data_dir
if args.dataset == "heart_disease":
    data_path = os.path.join(project_abspath, dict["dataset_dir"])
else:
    data_path = os.path.join(project_abspath, dict["dataset_dir"][f"{args.data_type}"]) 
    print(data_path)

rawdata = RawClass(data_path=data_path)
training_dls, test_dls = [], []


sorted_labels = []

for i in range(NUM_CLIENTS):
    g = torch.Generator()
    seed = 1000+i
    g.manual_seed(seed)

    
    train_ds = FedClass(rawdata=rawdata, center=i, train=True)
    test_ds = FedClass(rawdata=rawdata, center=i, train=False)
    
    training_dls.append(DataLoader(train_ds,batch_size=BATCH_SIZE,shuffle=True,generator=g))
    test_dls.append(DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False))

    label_counts = Counter(train_ds.labels)
    print(f"\nClient {i} label distribution:")
    print(sorted(label_counts.items()))

    for idx, label in enumerate(train_ds.labels):
            sorted_labels.append((label, i, idx))
    #sorted_labels.extend([(label, i) for label in train_ds.labels])


# list of labels across all clients with entry (label, client index, sample index)
sorted_labels.sort(key= lambda x: x[0])
    
# creates Pooled dataset with data from all clients
pooled_test_ds = FedClass(rawdata, pooled=True, train=False)
pooled_test_dl = DataLoader(pooled_test_ds, batch_size=BATCH_SIZE)


""" Prepare model and loss """
# We set model and dataloaders to be the same for each rep
global_init = BaselineModel.to(device)
criterion = BaselineLoss()

current_args = {
    "training_dataloaders": training_dls,
    "test_dataloaders": test_dls,
    "loss": criterion,
    "optimizer_class": Optimizer,
    "learning_rate": LR_DP,
    "num_steps": NUM_STEPS,
    "num_rounds": NUM_ROUNDS,
    "client_rate": CLIENT_RATE,
    "device": device,
    "metric": metric,
    "seed": args.seed
}



""" Prepare personalized epsilons """
# different distributions & different settings
SETTINGS = dict["rpdpfedavg"]["settings"]
MIN_EPSILON, MAX_EPSILON = dict["rpdpfedavg"]["min_epsilon"], dict["rpdpfedavg"]["max_epsilon"]
BoundedFunc = lambda values: np.array([min(max(x, MIN_EPSILON), MAX_EPSILON) for x in values])
epsilons = []
for name in GENERATE_EPSILONS_FUNC.keys():
    epsilons.extend([f"{name}-{_}" for _ in range(len(SETTINGS[name]))])


for ename in epsilons[:1]:
    name, p_id = ename.split('-')
# Generates list of epsilons w length = # of total samples
    full_epsilons = np.array(BoundedFunc(GENERATE_EPSILONS_FUNC[name](len(sorted_labels), SETTINGS[name][int(p_id)])))
    full_epsilons.sort()


    client_indices = [[idx for idx in range(len(sorted_labels)) if sorted_labels[idx][1] == i] for i in range(NUM_CLIENTS)]


    original_indices = [[sorted_labels[idx][2] for idx in range(len(sorted_labels)) if sorted_labels[idx][1]==i] for i in range(NUM_CLIENTS)]
    target_epsilons= [full_epsilons[client_indices[i]] for i in range(NUM_CLIENTS)]
    # target_epsilons =  np.array([[e for _, e in sorted(zip(original_indices[i], target_epsilons[i]))] for i in range(NUM_CLIENTS)])
    target_epsilons = [[e for _, e in sorted(zip(original_indices[i], target_epsilons[i]))] for i in range(NUM_CLIENTS)]

    for i, eps in enumerate(target_epsilons):
        print(f"Client {i}: avg epsilon = {np.mean(eps):.4f}")


    # We run Ditto with rPDP
    print(f" We run FedAvg with rPDP ({ename}) ...")
    set_random_seed(args.seed)
    privacy_engine = PrivacyEngine(accountant="fed_rdp", n_clients=NUM_CLIENTS)
    privacy_engine.prepare_fedrpdp(
        num_steps = NUM_STEPS,
        num_rounds = NUM_ROUNDS,
        client_rate = CLIENT_RATE,
        target_epsilons = target_epsilons,
        target_delta = TARGET_DELTA,
        max_epsilon = MAX_EPSILON,
        max_grad_norm = MAX_GRAD_NORM,
        max_physical_batch_size = MAX_PHYSICAL_BATCH_SIZE
    )
    current_args["model"] = copy.deepcopy(global_init)
    current_args["privacy_engine"] = privacy_engine
    current_args["reg_param"] = args.reg_param
    current_args["num_personal_steps"] = args.num_personal_steps

    current_args["pooled_test_dataloader"] = pooled_test_dl
    print("learning rate = ", current_args["learning_rate"])


    s = Ditto(**current_args, log=False)
    cm, perf_global, perf_personal, perf_pooled = s.run()
   

    expected_batch_size = [int(sum(acct.sample_rate)) for acct in s.privacy_engine.accountant.accountants]

    perf_data = {
        "global": perf_global,
        "personal": perf_personal,
        "pooled": perf_pooled
    }

# Save records for global, personal, and personal pooled acurracy 
    for key in ["global", "personal", "pooled"]:
        perf = perf_data[key]
        mean_perf = np.mean(perf[-3:])
        
        print(f"Mean performance of ditto rpdp {key}, Perf={mean_perf:.4f}, seed={args.seed}")
        
        record = [{ 
        "type": key,
        "lambda": args.reg_param,
        "num_personal_steps": args.num_personal_steps,
        "perf": str(perf),
        "mean_perf": round(mean_perf, 4),
        "e": f"{ename}-SCF",
        "d": TARGET_DELTA,
        "nm": round(s.privacy_engine.default_noise_multiplier, 2),
        "norm": MAX_GRAD_NORM,
        "bs": expected_batch_size[0],
        "seed": args.seed
        }]
        results_df = pd.DataFrame.from_dict(record)
        
        # Append to CSV
        write_header = not os.path.exists(save_file)
        results_df.to_csv(save_file, mode='a', index=False, header=write_header)

    del privacy_engine, s, cm
    torch.cuda.empty_cache()  # clears GPU cache



"""
    # We run Ditto with rPDP (StrongForAll)



    set_random_seed(args.seed)
    min_epsilon = min([min(per_client_epsilons) for per_client_epsilons in target_epsilons])
    print(min_epsilon)

    privacy_engine = PrivacyEngine(accountant="fed_rdp", n_clients=NUM_CLIENTS)
    privacy_engine.prepare_feddp(
        num_steps = NUM_STEPS,
        num_rounds = NUM_ROUNDS,
        sample_rate = BATCH_SIZE / min([len(train_dl.dataset) for train_dl in training_dls]),
        client_rate = CLIENT_RATE,
        target_epsilon = min_epsilon,
        target_delta = TARGET_DELTA,
        max_grad_norm = MAX_GRAD_NORM,
        max_physical_batch_size = MAX_PHYSICAL_BATCH_SIZE
    )
    current_args["model"] = copy.deepcopy(global_init)
    current_args["privacy_engine"] = privacy_engine
    current_args["reg_param"] = args.reg_param
    current_args["num_personal_steps"] = args.num_personal_steps
    current_args["pooled_test_dataloader"] = pooled_test_dl
    print("learning rate = ", current_args["learning_rate"])



    s = Ditto(**current_args, log=False)
    cm, perf_global, perf_personal, perf_pooled = s.run()

    expected_batch_size = [int(acct.sample_rate * len(train_dl.dataset)) for acct, train_dl in zip(s.privacy_engine.accountant.accountants, training_dls)]


    perf_data = {
        "global": perf_global,
        "personal": perf_personal,
        "pooled": perf_pooled
    }


    # Save records for global, personal, and personal pooled acurracy 
    for key in ["global", "personal", "pooled"]:
        perf = perf_data[key]
        mean_perf = np.mean(perf[-3:])
        

        print(f"Mean performance of StrongForAll {key}, eps={min_epsilon}, delta={TARGET_DELTA}, Perf={mean_perf:.4f}, seed={args.seed}")

        
        record = [{ 
        "type": key,
        "lambda": args.reg_param,
        "num_personal_steps": args.num_personal_steps,
        "perf": str(perf),
        "mean_perf": round(mean_perf, 4),
        "e": f"{ename}-StrongForAll",
        "d": TARGET_DELTA,
        "nm": round(s.privacy_engine.default_noise_multiplier, 2),
        "norm": MAX_GRAD_NORM,
        "bs": expected_batch_size[0],
        "seed": args.seed
        }]
        results_df = pd.DataFrame.from_dict(record)
        
        # Append to CSV
        write_header = not os.path.exists(save_file)
        results_df.to_csv(save_file, mode='a', index=False, header=write_header)

    del privacy_engine, s, cm
    torch.cuda.empty_cache()  # clears GPU cache







    # We run FedAvg with rPDP (Dropout)
    print(" We run FedAvg with rPDP (Dropout) ...")


    set_random_seed(args.seed)
    temp_epsilons = copy.deepcopy(target_epsilons)
    
    #DIFFERENCE BETWEEN METHODS
    #DROPS LOW EPSILON CLIENTS
    mean_epsilon = np.mean([np.mean(per_client_epsilons) for per_client_epsilons in temp_epsilons])
    for i in range(NUM_CLIENTS):
        mask = temp_epsilons[i] < mean_epsilon
        temp_epsilons[i][mask] = 0
        temp_epsilons[i][~mask] = mean_epsilon
        
    privacy_engine = PrivacyEngine(accountant="fed_rdp", n_clients=NUM_CLIENTS)
    privacy_engine.prepare_fedrpdp(
        num_steps = NUM_STEPS,
        num_rounds = NUM_ROUNDS,
        client_rate = CLIENT_RATE,
        target_epsilons = temp_epsilons,
        target_delta = TARGET_DELTA,
        max_epsilon = MAX_EPSILON,
        max_grad_norm = MAX_GRAD_NORM,
        max_physical_batch_size = MAX_PHYSICAL_BATCH_SIZE
    )
    
    current_args["model"] = copy.deepcopy(global_init)
    current_args["privacy_engine"] = privacy_engine
    current_args["reg_param"] = args.reg_param
    current_args["num_personal_steps"] = args.num_personal_steps
    current_args["pooled_test_dataloader"] = pooled_test_dl
    print("learning rate = ", current_args["learning_rate"])

    s = Ditto(**current_args, log=False)
    cm, perf_global, perf_personal, perf_pooled = s.run()
    
    expected_batch_size = [int(sum(acct.sample_rate)) for acct in s.privacy_engine.accountant.accountants]

    perf_data = {
        "global": perf_global,
        "personal": perf_personal,
        "pooled": perf_pooled
    }

    # Save records for global, personal, and personal pooled acurracy 
    for key in ["global", "personal", "pooled"]:
        perf = perf_data[key]
        mean_perf = np.mean(perf[-3:])
        
        print(f"Mean performance of ditto rpdp Dropout {key},  min_eps={min(target_epsilons[0]):.4f}, max_eps={max(target_epsilons[0]):.4f}, delta={TARGET_DELTA}, Perf={mean_perf:.4f}, seed={args.seed}")
        
        record = [{ 
        "type": key,
        "lambda": args.reg_param,
        "num_personal_steps": args.num_personal_steps,
        "perf": str(perf),
        "mean_perf": round(mean_perf, 4),
        "e": f"{ename}-Dropout",
        "d": TARGET_DELTA,
        "nm": round(s.privacy_engine.default_noise_multiplier, 2),
        "norm": MAX_GRAD_NORM,
        "bs": expected_batch_size[0],
        "seed": args.seed
        }]
        results_df = pd.DataFrame.from_dict(record)
        
        # Append to CSV
        write_header = not os.path.exists(save_file)
        results_df.to_csv(save_file, mode='a', index=False, header=write_header)

    del privacy_engine, s, cm
    torch.cuda.empty_cache()  # clears GPU cache












    print(f"Mean global performance of {name}, min_eps={min(target_epsilons[0]):.4f}, max_eps={max(target_epsilons[0]):.4f}, delta={TARGET_DELTA}, Perf={mean_perf_global:.4f}, seed={args.seed}")
    print(f"Mean personal performance of {name}, min_eps={min(target_epsilons[0]):.4f}, max_eps={max(target_epsilons[0]):.4f}, delta={TARGET_DELTA}, Perf={mean_perf_personal:.4f}, seed={args.seed}")
    print(f"Mean pooled personal performance of {name}, min_eps={min(target_epsilons[0]):.4f}, max_eps={max(target_epsilons[0]):.4f}, delta={TARGET_DELTA}, Perf={mean_perf_pooled:.4f}, seed={args.seed}")


    # prepare global results
    record_global = [{
        "perf": str(perf_global),  # store as string
        "mean_perf": round(np.mean(perf_global[-3:]), 4),
        "e": f"{ename}-Dropout", 
        "d": TARGET_DELTA,
        "nm": round(s.privacy_engine.default_noise_multiplier, 2),
        "norm": MAX_GRAD_NORM,
        "bs": expected_batch_size[0], 
        "seed": args.seed
    }]

    # Prepare personal results
    record_personal = [{
        "perf": str(perf_personal),
        "mean_perf": round(np.mean(perf_personal[-3:]), 4),
        "e": f"{ename}-Dropout", 
        "d": TARGET_DELTA,
        "nm": round(s.privacy_engine.default_noise_multiplier, 2),
        "norm": MAX_GRAD_NORM,
        "bs": expected_batch_size[0], 
        "seed": args.seed
    }]
    
     # Prepare personal results
    record_pooled = [{
        "perf": str(perf_pooled),
        "mean_perf": round(np.mean(perf_pooled[-3:]), 4),
        "e": f"{ename}-Dropout", 
        "d": TARGET_DELTA,
        "nm": round(s.privacy_engine.default_noise_multiplier, 2),
        "norm": MAX_GRAD_NORM,
        "bs": expected_batch_size[0], 
        "seed": args.seed
    }]
    
    
    results_global = pd.DataFrame.from_dict(record_global)
    results_personal = pd.DataFrame.from_dict(record_personal)
    results_pooled = pd.DataFrame.from_dict(record_pooled)

    
    
    results_global.to_csv(save_filename_global, mode='a', index=False,header=False)
    results_personal.to_csv(save_filename_personal, mode='a', index=False,header=False)
    results_pooled.to_csv(save_filename_pooled, mode='a', index=False,header=False)

    
    del privacy_engine, s, cm, mean_perf_personal, mean_perf_global, mean_perf_pooled
    """