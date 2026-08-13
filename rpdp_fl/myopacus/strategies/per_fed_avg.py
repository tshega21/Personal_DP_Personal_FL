import numpy as np
import time
import torch
from typing import List
import copy
import random

from myopacus import PrivacyEngine
from myopacus.strategies.strategies_utils import _Model


#def set_random_seed(seed_value):
    #np.random.seed(seed_value)
    #torch.manual_seed(seed_value)
    #torch.cuda.manual_seed(seed_value)
    
def set_random_seed(seed_value):
    
    random.seed(seed_value)
    np.random.seed(seed_value)

    #PyTorch CPU
    torch.manual_seed(seed_value)
    #PyTorch GPU 
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    #torch.use_deterministic_algorithms(True)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.set_num_threads(1)
      
def evaluate_model_on_tests(
    models_list, pooled_dl = None, return_pred=False
):
    """This function takes a pytorch model and evaluate it on a list of
    dataloaders using the provided metric function.
    Parameters
    ----------
    models_list: List[torch.nn.Module],
        A trained model that can forward the test_dataloaders outputs

    Returns
    -------
    dict
        A dictionnary with keys client_test_{0} to 
        client_test_{len(test_dataloaders) - 1} and associated scalar metrics 
        as leaves.
    """
    
    results_dict = {}
    y_true_dict = {}
    y_pred_dict = {}
    
    with torch.no_grad():        

        for _model in models_list:
            
            _model.model.to(_model._device).eval()
            if pooled_dl is not None:
                test_dataloader_iterator = iter(pooled_dl)
            else: 
                test_dataloader_iterator = iter(_model._test_dl)
                
            y_pred_final = []
            y_true_final = []
            
            for batch in test_dataloader_iterator:
                batch = tuple(t.to(_model._device) for t in batch)
                if len(batch) == 2: # for other datasets
                    logits = _model.model(batch[0])
                    loss = _model._loss(logits, batch[1])

                elif len(batch) == 4: # for snli dataset
                    inputs = {'input_ids':    batch[0],
                                'attention_mask': batch[1],
                                'token_type_ids': batch[2],
                                'labels':         batch[3]}
                    outputs = _model.model(**inputs) # output = loss, logits, hidden_states, attentions
                    loss, logits = outputs[:2]
                
                y_pred_final.append(logits.detach().cpu().numpy())
                y_true_final.append(batch[1].detach().cpu().numpy())
            

            y_true_final = np.concatenate(y_true_final)
            y_pred_final = np.concatenate(y_pred_final)
            
            correct = _model._metric(y_true=y_true_final, y_pred=y_pred_final)
            results_dict[f"client_test_{_model.client_id}"] = correct
            #print(f"Client {_model.client_id}:\t {correct} / {len(y_true_final)}")
            
            if return_pred:
                y_true_dict[f"client_test_{_model.client_id}"] = y_true_final
                y_pred_dict[f"client_test_{_model.client_id}"] = y_pred_final
                
    if return_pred:
        return results_dict, y_true_dict, y_pred_dict
    else:
        return results_dict
    

class Ditto:
    """Federated Averaging Strategy class.

    The Federated Averaging strategy is the most simple centralized FL strategy.
    Each client first trains his version of a global model locally on its data,
    the states of the model of each client are then weighted-averaged and returned
    to each client for further training.

    References
    ----------
    - https://arxiv.org/abs/1602.05629

    Parameters
    ----------
    training_dataloaders : List
        The list of training dataloaders from multiple training centers.
    model : torch.nn.Module
        An initialized torch model.
    loss : torch.nn.modules.loss._Loss
        The loss to minimize between the predictions of the model and the
        ground truth.
    optimizer_class : torch.optim.Optimizer
        The class of the torch model optimizer to use at each step.
    learning_rate : float
        The learning rate to be given to the optimizer_class.
    num_steps : int
        The number of steps to do on each client at each round.
    num_rounds : int
        The number of communication rounds to do.
    log: bool, optional
        Whether or not to store logs in tensorboard. Defaults to False.
    log_period: int, optional
        If log is True then log the loss every log_period batch updates.
        Defauts to 100.
    bits_counting_function : Union[callable, None], optional
        A function making sure exchanges respect the rules, this function
        can be obtained by decorating check_exchange_compliance in
        flamby.utils. Should have the signature List[Tensor] -> int.
        Defaults to None.
    logdir: str, optional
        Where logs are stored. Defaults to ./runs.
    log_basename: str, optional
        The basename of the created log_file. Defaults to fed_avg.
    """

    def __init__(
        self,
        training_dataloaders: List, 
        test_dataloaders: List, # added by Junxu
        # Dataloader to test over all clients
        pooled_test_dataloader:torch.utils.data.DataLoader, 
        model: torch.nn.Module,
        loss: torch.nn.modules.loss._Loss,
        metric: callable,
        optimizer_class: torch.optim.Optimizer,
        learning_rate: float,
        client_rate: float,
        num_steps: int,
        num_rounds: int,
        num_personal_steps: int,
        reg_param: float = 1, #regularization parameter
        privacy_engine: PrivacyEngine = None,
        privacy_engine_ditto: PrivacyEngine = None,
        device: str = "cuda:0",
        log: bool = False,
        log_period: int = 100,
        bits_counting_function: callable = None,
        logdir: str = "./runs",
        log_basename: str = "fed_avg",
        seed: int = None,
        **kwargs
    ):
        self.privacy_engine = privacy_engine
        self.privacy_engine_ditto = privacy_engine_ditto
        self.reg_param = reg_param
        self.client_rate = client_rate
        self.num_rounds = num_rounds
        self.num_steps = num_steps
        self.pooled_test_dataloader = pooled_test_dataloader
        
        #do I add this or no? how to include this in DP accounting :/
        self.num_steps = num_steps
        self.num_personal_steps = num_personal_steps
        
        self.log = log
        self.log_period = log_period
        self.log_basename = log_basename
        self.logdir = logdir
        self._seed = seed
        set_random_seed(self._seed)
        self.rng = np.random.default_rng(seed)

        self.local_models_list = [
            _Model(
                #don't need deepcopy because deepcopy is done in strategies_utils.py
                model=model,
                optimizer_class=optimizer_class,
                lr=learning_rate,
                train_dl=_train_dl,
                test_dl=_test_dl,
                device=device,
                metric=metric,
                loss=loss,
                log=self.log,
                client_id=i,
                log_period=self.log_period,
                log_basename=self.log_basename,
                logdir=self.logdir,
                seed=self._seed,
            )
            for i, (_train_dl, _test_dl) in enumerate(list(zip(training_dataloaders, test_dataloaders)))
        ]
        
        set_random_seed(self._seed)
        self.personal_models_list = [
            _Model(
                model=model,
                optimizer_class=optimizer_class,
                lr=learning_rate,
                train_dl=_train_dl,
                test_dl=_test_dl,
                device=device,
                metric=metric,
                loss=loss,
                log=self.log,
                client_id=i,
                log_period=self.log_period,
                log_basename=self.log_basename,
                logdir=self.logdir,
                seed=self._seed,
            )
            for i, (_train_dl, _test_dl) in enumerate(list(zip(training_dataloaders, test_dataloaders)))
        ]

        if self.privacy_engine is not None:
            assert (self.privacy_engine.accountant.mechanism() == "idp"), \
                 "DataType of `privacy_engine.accountant` must be `IndividualAccountant` in FL setup."
            
            for l_model in self.local_models_list:
                l_model._make_private(self.privacy_engine)
        if self.privacy_engine_ditto is not None: 
            assert (self.privacy_engine_ditto.accountant.mechanism() == "idp"), \
                "DataType of `privacy_engine.accountant` must be `IndividualAccountant` in FL setup."
            for p_model in self.personal_models_list:
                p_model._make_private(self.privacy_engine_ditto)


        self.num_clients = len(training_dataloaders)
        self.bits_counting_function = bits_counting_function

    


    def _local_optimization(self, _model: _Model):
        """Carry out the local optimization step."""
        if self.privacy_engine is None:
            _model._local_train(self.num_steps)
            
        # privacy engine exists that is not idp (multiple accountants)
        elif not (self.privacy_engine.accountant.mechanism() == "idp"):
            _model._local_train(self.num_steps, \
                                privacy_accountant=self.privacy_engine.accountant)
            
        # every client has their own privacy accountant
        else:
            _model._local_train(self.num_steps, \
                                privacy_accountant=self.privacy_engine.accountant.accountants[_model.client_id])


    def _personal_optimization(self, personal_model: _Model, global_params: List[torch.Tensor]):
        """Carry out the local optimization step."""
        if self.privacy_engine_ditto is None:
            personal_model._ditto_local_train(global_params,\
                                            self.num_personal_steps, self.reg_param)
            
        # privacy engine exists that is not idp (multiple accountants)
        elif not (self.privacy_engine_ditto.accountant.mechanism() == "idp"):
            personal_model._ditto_local_train(global_params, \
                                self.num_personal_steps, self.reg_param, \
                                privacy_accountant=self.privacy_engine_ditto.accountant)
            
        # every client has their own privacy accountant
        else:
            personal_model._ditto_local_train(global_params, \
                               self.num_personal_steps, self.reg_param,\
                               privacy_accountant=self.privacy_engine_ditto.accountant.accountants[personal_model.client_id] )
    """
    def compare_models(self, state_dict1,state_dict2):
        models_differ = 0
        for key_item_1, key_item_2 in zip(state_dict1.items(), state_dict2.items()):
            if torch.equal(key_item_1[1], key_item_2[1]):
                pass
            else:
                models_differ += 1
                if (key_item_1[0] == key_item_2[0]):
                    # print('Mismtach found at', key_item_1[0])
                    pass
                else:
                    raise Exception
        if models_differ == 0:
            print('Models match perfectly! :)')

    def detach_state_dict(self, model: torch.nn.Module):
        '''
        Returns a detached copy of the model's state_dict.
        All tensors are detached and moved to CPU.
        '''
        detached_dict = {}
        for key, value in model.state_dict().items():
            detached_dict[key] = value.detach().cpu().clone()
        return detached_dict
    """
    def perform_round(self):
        """Does a single federated averaging round. The following steps will be
        performed:

        - each model will be trained locally for num_steps batches.
        - the parameter updates will be collected and averaged. Averages will be
          weighted by the number of samples in each client
        - the averaged updates will be used to update the local model
        
        Global round
        """
        local_updates = list()
        total_number_of_samples = 0
        selected_idx_client = []
        
        
        while len(selected_idx_client) == 0:
            
            # boolean mask that samples according to client_rate
            mask = self.rng.random(self.num_clients) < self.client_rate
            selected_idx_client = np.where(mask)[0]
            if len(selected_idx_client) == 0:
                selected_idx_client = np.array([self.rng.integers(self.num_clients)])

            selected_idx_client = np.where(mask == True)[0]
            #print("selected_idx_client: ", selected_idx_client)
        
        model_lists = list(zip(self.local_models_list, self.personal_models_list))
        
        selected_models = [model_lists[idx] for idx in selected_idx_client]
        #local training round for every client 
        for local_model, personal_model in selected_models:
            

            #print(f"Client {local_model.client_id} ...")
            # Local Optimization
            _local_previous_state = local_model._get_current_params()
            
            global_snapshot = [p.detach().clone() for p in local_model.model.parameters()]
             #calls personalization on w_k initial global model


            # calls local_train from strategies_utils.py for num of local steps
            self._local_optimization(local_model)
            _local_next_state = local_model._get_current_params()


            self._personal_optimization(personal_model, global_snapshot)            
            

            
            # Recovering updates (w^t_k - w^t), how much params change after all local steps
            updates = [
                new - old for new, old in zip(_local_next_state, _local_previous_state)
            ]

            #deletes copy of params
            del _local_next_state

            # Reset local model
            for p_new, p_old in zip(local_model.model.parameters(), _local_previous_state):
                p_new.data = torch.from_numpy(p_old).to(p_new.device)
            del _local_previous_state

            if self.bits_counting_function is not None:
                self.bits_counting_function(updates)
            
            # list of updates and update number of samples 

            local_updates.append({"updates": updates, "n_samples": len(local_model._train_dl.dataset)})
            total_number_of_samples += len(local_model._train_dl.dataset)

        # Aggregation step
        
        aggregated_delta_weights = [
            None for _ in range(len(local_updates[0]["updates"]))
        ]
        
        # iterate through every parameter and weight
        for idx_weight in range(len(local_updates[0]["updates"])):
            aggregated_delta_weights[idx_weight] = sum(
                [
                    local_updates[idx_client]["updates"][idx_weight]
                    * local_updates[idx_client]["n_samples"]
                    for idx_client in range(len(selected_idx_client))
                ]
            )
            #weighted average
            aggregated_delta_weights[idx_weight] /= float(total_number_of_samples)

        # reset local model to new global model
        for _model in self.local_models_list:
            _model._update_params(aggregated_delta_weights)

    
    # def run(self, metric, device):
    def run(self):
        """This method performs self.nrounds rounds of averaging
        and returns the list of models.
        """
        def compute_per_client_accuracy(perf_dict, y_true_dict):
            per_client_acc = {}

            for k in perf_dict:
                per_client_acc[k] = round(perf_dict[k] / len(y_true_dict[k]),4)

            return per_client_acc
        
        all_round_results_global = []
        all_round_results_personal = []
        all_round_results_personal_pooled = []

        #seed_gen = torch.Generator()
        #seed_gen.manual_seed(123)  # master seed
        
        if self.privacy_engine:
                ret = self.privacy_engine.accountant.get_epsilon(delta=self.privacy_engine.target_delta, mode="max")
                print("initial privacy cost of all clients: ", ret)

        for r in range(self.num_rounds):
            self.perform_round()
            #seed = torch.randint(0, 2**31, (1,), generator=seed_gen).item()

            
            perf_global, y_true_dict1, y_pred_dict1 = evaluate_model_on_tests(self.local_models_list, return_pred=True)
            perf_personal, y_true_dict2, y_pred_dict2 = evaluate_model_on_tests(self.personal_models_list,  return_pred=True)
            perf_pooled, y_true_dict3, y_pred_dict3 = evaluate_model_on_tests(self.personal_models_list, pooled_dl= self.pooled_test_dataloader, return_pred=True)
            
            if self.privacy_engine:
                ret = self.privacy_engine.accountant.get_epsilon(delta=self.privacy_engine.target_delta, mode="max")
                #print("current privacy cost of all clients: ", ret)

            correct_global = np.array( [v for _, v in perf_global.items()] ).sum()
            total_global = np.array( [len(v) for _, v in y_true_dict1.items()] ).sum()
            
            
            correct_personal = np.array(  [v for _, v in perf_personal.items()]).sum()
            total_personal = np.array(  [len(v) for _, v in y_true_dict2.items()] ).sum()
            
            correct_personal_pooled = np.array(  [v for _, v in perf_pooled.items()]).sum()
            total_personal_pooled = np.array(  [len(v) for _, v in y_true_dict3.items()] ).sum()
            
            per_client_acc_g = compute_per_client_accuracy(perf_global, y_true_dict1)
            per_client_acc_p = compute_per_client_accuracy(perf_personal, y_true_dict2)
            per_client_acc_pp = compute_per_client_accuracy(perf_pooled, y_true_dict3)

           
            print(f"Round={r}")
            # make an accuracy variable here instead of doing something 3 times 
            if r == (self.num_rounds -1):
                if self.privacy_engine:
                    ret = self.privacy_engine.accountant.get_epsilon(delta=self.privacy_engine.target_delta, mode="max")
                    print("current privacy cost of all clients: ", ret)
                print(f"Round={r}, per client global perf={list(per_client_acc_g.values())}, mean perf={correct_global}/{total_global} ({correct_global/total_global:.4f}%)")
                print(f"Round={r}, per client personal perf={list(per_client_acc_p.values())}, mean perf={correct_personal}/{total_personal} ({correct_personal/total_personal:.4f}%)")
                print(f"Round={r}, per client personal pooled={list(per_client_acc_pp.values())}, mean perf={correct_personal_pooled}/{total_personal_pooled} ({correct_personal_pooled/total_personal_pooled:.4f}%)")

            all_round_results_global.append(round(correct_global/total_global, 4))
            all_round_results_personal.append(round(correct_personal/total_personal, 4))
            all_round_results_personal_pooled.append(round(correct_personal_pooled/total_personal_pooled, 4))

            


        return [m.model for m in self.local_models_list], all_round_results_global, all_round_results_personal, all_round_results_personal_pooled
