from agent import buying_mlpAgent
from auction import data_purchase_classification_competition
from data import tabular_dataset
from user import *
import argh, os, pickle, copy, itertools
import numpy as np

N_EPISODE=1 # One may want to change this part.
N_BUYERS=18

# Insurance
def construct_insurance_agents(n_budget=200, n_seed=50):
    uncertain_buyer = dict()
    uncertain_buyer['class'] = buying_mlpAgent
    uncertain_buyer['args'] = {'x_dim':16, 'n_class':2, 'epoch':10, 
                                'n_layers':0, 'lr':5e-3, 'task':"C", 'hidden':64,
                                'retrain_limit':50, 'bsize':16, 'retrain_max':10*10**3,
                                'n_budget': n_budget, 'AL_strategy':'uncertainty',
                                'AL_parameter': 0.3}
    uncertain_buyer['init_ds'] = n_seed

    return uncertain_buyer

def generate_insurance_exp_run(expno_name, n_budget, n_seed, max_iter, n_episode, noisy_prop=0.3):    
    alpha=4.0
    uncertain_buyer=construct_insurance_agents(n_budget, n_seed)

    #EXP CONFIGS
    exp = dict()
    exp['expno'] = expno_name
    exp['n_runs'] = n_episode
    
    #RUN CONFIGS
    runs = []
    r_id = 0
    run_temp = dict()
    run_temp['use-gpu'] = False
    run_temp['auction'] = data_purchase_classification_competition
    run_temp['auction_args'] = {'c':0, 'r':1}
    run_temp['dataset'] = tabular_dataset
    run_temp['dataset_init'] = tabular_dataset
    run_temp['users'] = []
    run_temp['dargs'] = {'path':'/home/users/yckwon/data/competing_AI/',
                        'csv_prefix':'insurance',
                        'noisy_prop':noisy_prop}
    run_temp['print-freq']=500
    run_temp['print-test-freq']=100
    run_temp['max-iters'] = max_iter
    run_temp['special-log'] = ['agg_corr','y_hats','bidder','test_pred']
        
    for seed in range(n_episode):
        run = copy.deepcopy(run_temp)
        run['auction_args']['alpha'] = alpha
        run['seed'] = seed
        run['agents'] = []
        for i in N_BUYERS:
            run['agents'].append(uncertain_buyer)
        run['r_id'] = r_id
        r_id += 1
        runs.append(run)

    return exp, runs 


def config001IY():
    expno_name, n_budget, n_seed, max_iter, n_episode = '001IY', 0, 100, 10000, N_EPISODE
    exp, runs = generate_insurance_exp_run(expno_name, n_budget, n_seed, max_iter, n_episode)
    return exp, runs 

def config002IY():
    expno_name, n_budget, n_seed, max_iter, n_episode = '002IY', 100, 100, 10000, N_EPISODE
    exp, runs = generate_insurance_exp_run(expno_name, n_budget, n_seed, max_iter, n_episode)
    return exp, runs     

def config003IY():
    expno_name, n_budget, n_seed, max_iter, n_episode = '003IY', 200, 100, 10000, N_EPISODE
    exp, runs = generate_insurance_exp_run(expno_name, n_budget, n_seed, max_iter, n_episode)
    return exp, runs 

def config004IY():
    expno_name, n_budget, n_seed, max_iter, n_episode = '004IY', 400, 100, 10000, N_EPISODE
    exp, runs = generate_insurance_exp_run(expno_name, n_budget, n_seed, max_iter, n_episode)
    return exp, runs      






