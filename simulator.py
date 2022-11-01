from competition import *
from analysis_utils import _y_hats, _agg_corr, _buys, _test_pred
import dill as pickle
import torch, torchvision
import torch.utils 
import numpy as np
import argh, warnings, random
from collections import defaultdict

parser= argh.ArghParser()

def setup_agents(config):
    _set_seed(config)
    agents = []
    for i,a in enumerate(config['agents']):
        agent = a['class'](**a['args'])
        agent.id = i  
        agents.append(agent)
    return agents

def setup_mechanism(config, agents, datastream):
    return config['competition'](agents,datastream, **config['competition_args'])
        
def init_agent_data(config, agents):
    # apply training on seed data, if needed
    for j,a in enumerate(agents):
        _set_seed(config,mod=j+1)

        #define config if not none
        if config['dataset_init'] is not None:
            data_loader = torch.utils.data.DataLoader(
                config['dataset_init'](**config['dargs']),
                batch_size=config['agents'][j]['init_ds'],
                shuffle=True)

            for _,batch in enumerate(data_loader): #only run once
                try:
                    x,y = batch
                except:
                    x,y,_ = batch
                a.add_data(x,y)
                break
            a._seed_train()          

def setup_datastream(config, is_train=True):
    _set_seed(config,mod=-config['seed'])
    try:
        print('Train and test are different')
        if is_train is True:
            return torch.utils.data.DataLoader(
                config['dataset'](is_train=True, **config['dargs']),
                batch_size=1,
                shuffle=True)
        else:
            return torch.utils.data.DataLoader(
                config['dataset'](is_train=False, **config['dargs']),
                batch_size=1,
                shuffle=True)
    except:
        print('Train and test are same')
        return torch.utils.data.DataLoader(
                config['dataset'](**config['dargs']),
                batch_size=1,
                shuffle=True)


def setup_log(config, agents):
    logger = defaultdict(list)
    #need to define lambdas in order to pickle
    logger['agents'] = defaultdict(lambda : defaultdict(list))
    logger['config'] = config
    
    for i,a in enumerate(agents):
        assert a.id == i, "Agent id mismatch in logging setup"
        
    return logger

def _set_seed(config,mod=0):
    print('Seed is not fixed')

def _special_log(config,logger):
    special_log = dict()
    if 'special-log' in config:
        for sl in config['special-log']:
            if sl == 'y_hats':
                special_log['y_hats'] = _y_hats(logger)
            elif sl == 'agg_corr':
                special_log['agg_corr'] = _agg_corr(logger)
            elif sl == 'bidder':
                special_log['bidder'] = _bids(logger)
            elif sl == 'test_pred':
                special_log['test_pred'] = _test_pred(logger)
            else:
                warnings.warn(f'Special logging of type {sl} not supported')
    
    market_share_list = logger['winner']
    special_log['market_share_list'] = market_share_list
    np.savez(open('results/special_log','wb'),**special_log) 
    return special_log


def _sim(config, comp, datastream, logger, test_datastream=None):
    _set_seed(config) 
    run_id = config['r_id'] if 'r_id' in config else ''

    if test_datastream is None:
        assert False, 'PLEASE CHECK TEST_DATASTREAM'

    logger['test_acc'] = []
    for i,data in enumerate(datastream):
        if i >= config['max-iters']:
            print(f'Run{run_id}: simulation complete')
            break
            
        comp.logger = logger 
        comp.run(data, logger)

        if i % config['print-freq'] == 0:
            print(f'Run{run_id}: At round {i}',flush=True)

    # LAST TRAIN
    comp._last_train()

    # TEST
    tmp_list = []
    for j, test_data in enumerate(test_datastream):
        if j >= 3000:
            break
        tmp = comp.score_test_models(test_data)
        tmp_list.append(tmp)
    tmp_list = np.array(tmp_list)
    print(f'{len(tmp_list)}, Marginal acc: {np.mean(tmp_list)}',flush=True)

def _print_result(agents):
    print('-'*30)
    print('Summary')
    print('-'*30)

    wins_list = []
    for j,a in enumerate(agents):
        print(f'AL strategy: {a.AL_strategy}')
        print(f'Initial budgets: {a.initial_budget}')
        print(f'Market share: {a.wins}')
        wins_list.append(a.wins)

def main(config):
    '''
    MAIN function
    '''
    agents = setup_agents(config)
    init_agent_data(config, agents)
    datastream = setup_datastream(config)
    test_datastream = setup_datastream(config, is_train=False)
    # datastream.users = setup_users(config)
    comp = setup_mechanism(config, agents, datastream)
    # setup_users_special(config, agents, datastream)
    logger = setup_log(config, agents)
    _sim(config, comp, datastream, logger, test_datastream)
    _special_log(config, logger)
    _print_result(agents)

parser.add_commands([main])
# dispatching 
if __name__ == '__main__':
    parser.dispatch()


