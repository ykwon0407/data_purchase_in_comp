import torch
import torch
import torch.nn.functional as F
import numpy as np
import pickle, glob, os, re, sys
import multiprocessing
from joblib import Parallel, delayed

sys.path.insert(0,'..')

from data import *
from agent import *
from auction import *
from user import *
from simulator import *

def _depickle(run, log=True):
    os.chdir(run)
    r_id = int(run.split('/')[-1][3:])
    logger = None
    if log:
        logger = pickle.load(open('logger.pickle','rb'))
    config = pickle.load(open('config.pickle','rb'))
    return config, logger

def _enumerate_runs(outpath):
    #also initializes the logs & configs
    runs = glob.glob(f'{outpath}/run*')
    logs = [None]*len(runs)
    configs = [None]*len(runs)
    return runs, logs, configs

def _sort_runs(runs):
    runs_sorted = [None]*len(runs)
    for run in runs:
        s = int(re.sub("[^0-9]", "", run.split('/')[-1]))
        runs_sorted[s] = run
    return runs_sorted

def _agg_corr(logger):
    return torch.tensor(logger['agg-correct']).int().numpy()

def _race_fair(logger):
    agg_corr = _agg_corr(logger)
    x = torch.stack(logger['x']).squeeze()
    return (x, agg_corr)

def _y_hats(logger):
    A = logger['agents']
    Y_hat = torch.zeros((1+len(A), len(A[0]['y_hat'])))
    for i in A:
        Y_hat[i] = torch.tensor(A[i]['y_hat']) 
    #last Y_hat row is ground truth
    Y_hat[-1] = torch.tensor(logger['y'])
    return Y_hat.numpy()

def _bids(logger):
    return torch.tensor(logger['bidder']).int().numpy()    

def _test_pred(logger):
    A = logger['agents']
    Y_test_hat = torch.zeros((1+len(A), len(A[0]['test_pred'])))
    for i in A:
        Y_test_hat[i] = torch.tensor(A[i]['test_pred']) 
    #last Y_hat row is ground truth
    Y_test_hat[-1] = torch.tensor(logger['test_y'])
    return Y_test_hat.numpy()

def _cf_process(logger):
    x = torch.tensor(logger['x'])
    u_choices = torch.tensor(logger['choices'])
    y_hats = []
    for i in logger['agents']:
        #print(i)
        y_hat = torch.tensor(logger['agents'][i]['y_hat'])
        y_hats.append(y_hat)
    y_hats = torch.stack(y_hats).t()
    choices = torch.nn.functional.one_hot(u_choices, i+1) 
    rec = torch.sum(y_hats * choices,dim=1)
    U = logger['uEmb']
    V = logger['vEmb']  
    return x, rec, U, V, u_choices, y_hats

def process_select(proc):
    if proc == 'agg_corr':
        process = _agg_corr
    elif proc == 'y_hats':
        process = _y_hats
    elif proc == 'cf':
        process = _cf_process
    elif proc == 'race_fair':
        process = _race_fair
    else:
        raise ValueError(f'Proc {proc} not supported')
    return process

def depickler(outpath, log=False):
    runs, logs, configs = _enumerate_runs(outpath)
    for run in runs:
        print(run)
        r_id = int(run.split('/')[-1][3:])
        config, logger = _depickle(run,log)
        configs[r_id] = config
        if logger != None:
            logs[r_id] = logger
    
    return configs, logs

def depickler_special(outpath):
    runs, logs, configs = _enumerate_runs(outpath)
    for run in runs:
        # print(run)
        r_id = int(run.split('/')[-1][3:])
        try:
            config, _ = _depickle(run,log=False)
            configs[r_id] = config
            logs[r_id] = np.load(f'{run}/special_log')
        except:
            pass
    
    return configs, logs

def depickler_process(outpath, proc='agg_corr'):
    runs, logs, configs = _enumerate_runs(outpath)
    for run in runs:
        print(run)
        r_id = int(run.split('/')[-1][3:])
        config, logger = _depickle(run,log=True)
        configs[r_id] = config
        logs[r_id] = process_select(proc)(logger) 
        del logger

    return configs, logs

def para_depickler(outpath, proc='agg_corr'):
    from joblib import Parallel, delayed
    import multiprocessing

    process = process_select(proc)  
    def processInput(run):
        config, logger = _depickle(run,log=True)
        log = process(logger) 
        del logger
        return config,log

    n_cores = multiprocessing.cpu_count()
    runs, logs, configs = _enumerate_runs(outpath)
    runs = _sort_runs(runs)
    results = Parallel(n_jobs=n_cores)(delayed(processInput)(r) for r in runs)
    configs,logs = zip(*results)
    
    return configs, logs

def data_saver(data, outpath, proc):
    process_select(proc) #just to make sure user gave valid proc
    np.save(f'{outpath}/{proc}',data)
