import numpy as np
import time
import torch
from typing import List

from myopacus import PrivacyEngine
from myopacus.strategies.strategies_utils import _Model

def set_random_seed(seed_value):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    
def evaluate_model_on_tests(
    models_list, return_pred=False
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
            test_dataloader_iterator = iter(_model._test_dl)
            y_pred_final = []
            y_true_final = []
            for batch in iter(test_dataloader_iterator):
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
                y_true_final.append(batch[-1].detach().cpu().numpy())

            y_true_final = np.concatenate(y_true_final)
            y_pred_final = np.concatenate(y_pred_final)
            
            correct = _model._metric(y_true=y_true_final, y_pred=y_pred_final)
            results_dict[f"client_test_{_model.client_id}"] = correct
            # print(f"Client {_model.client_id}:\t {correct} / {len(y_true_final)}")
            
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
        self.reg_param = reg_param
        self.num_personal_steps = num_personal_steps
        self.client_rate = client_rate
        self.num_rounds = num_rounds
        self.num_steps = num_steps
        
        #do I add this or no? how to include this in DP accounting :/
        self.num_personal_steps = num_personal_steps
        
        self.log = log
        self.log_period = log_period
        self.log_basename = log_basename
        self.logdir = logdir
        self._seed = seed
        set_random_seed(self._seed)

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
            
            for _model in self.models_list:
                _model._make_private(self.privacy_engine)

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


    def _personal_optimization(self,local_model: _Model, personal_model: _Model):
        """Carry out the local optimization step."""
        if self.privacy_engine is None:
            local_model._ditto_local_train(personal_model,\
                                    self.num_personal_steps, self.reg_param)
            
        # privacy engine exists that is not idp (multiple accountants)
        elif not (self.privacy_engine.accountant.mechanism() == "idp"):
            local_model._ditto_local_train(personal_model, \
                                self.num_personal_steps, self.reg_param, \
                                privacy_accountant=self.privacy_engine.accountant)
            
        # every client has their own privacy accountant
        #DO I ATTACH PRIVACY ENGINE TO LOCAL OR PERSONAL ***
        else:
            local_model._ditto_local_train(personal_model, \
                               self.num_personal_steps, self.reg_param, \
                                privacy_accountant=self.privacy_engine.accountant.accountants[_model.client_id])


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
            mask = (np.random.random(self.num_clients) < self.client_rate)
            selected_idx_client = np.where(mask == True)[0]
            print("selected_idx_client: ", selected_idx_client)
        
        model_lists = list(zip(self.local_models_list, self.personal_models_list))
        
        selected_models = [model_lists[idx] for idx in selected_idx_client]
        #local training round for every client 
        for local_model, personal_model in selected_models:
            print(f"Client {local_model.client_id} ...")
            # Local Optimization
            _local_previous_state = local_model._get_current_params()
            
            # calls local_train from strategies_utils.py for num of local steps
            self._local_optimization(local_model)
            _local_next_state = local_model._get_current_params()
            
            self._personal_optimization(local_model, personal_model)

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
        all_round_results_global = []
        all_round_results_personal = []

        for r in range(self.num_rounds):
            self.perform_round()
            perf_global, y_true_dict1, y_pred_dict1 = evaluate_model_on_tests(self.local_models_list, return_pred=True)
            perf_personal, y_true_dict2, y_pred_dict2 = evaluate_model_on_tests(self.personal_models_list, return_pred=True)

            if self.privacy_engine:
                ret = self.privacy_engine.accountant.get_epsilon(delta=self.privacy_engine.target_delta, mode="max")
                print("current privacy cost of all clients: ", ret)

            correct_global = np.array(
                [v for _, v in perf_global.items()]
            ).sum()
            total = np.array(
                [len(v) for _, v in y_true_dict1.items()]
            ).sum()
            
            correct_personal = np.array(
                [v for _, v in perf_personal.items()]
            ).sum()
            total = np.array(
                [len(v) for _, v in y_true_dict2.items()]
            ).sum()
            
            print(f"Round={r}, global perf={list(perf_global.values())}, mean perf={correct_global}/{total} ({correct_global/total:.4f}%)")
            print(f"Round={r}, personal perf={list(perf_personal.values())}, mean perf={correct_personal}/{total} ({correct_personal/total:.4f}%)")

            all_round_results_global.append(round(correct_global/total, 4))
            all_round_results_personal.append(round(correct_personal/total, 4))


        return [m.model for m in self.local_models_list], all_round_results_global, all_round_results_personal
