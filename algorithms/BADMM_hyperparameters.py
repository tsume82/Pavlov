import tensorflow as tf

import os.path
from datetime import datetime
import numpy as np
from cma import bbobbenchmarks as bn
from algorithms.GPS.source.gps import __file__ as gps_filepath
from algorithms.GPS.source.gps.agent.lto.agent_cmaes import AgentCMAES
from algorithms.GPS.source.gps.agent.lto.cmaes_world import CMAESWorld
from algorithms.GPS.source.gps.algorithm.algorithm import Algorithm
from algorithms.GPS.source.gps.algorithm.cost.cost import Cost
from algorithms.GPS.source.gps.algorithm.dynamics.dynamics_lr_prior import DynamicsLRPrior
from algorithms.GPS.source.gps.algorithm.dynamics.dynamics_prior_gmm import DynamicsPriorGMM
from algorithms.GPS.source.gps.algorithm.policy.policy_prior_gmm import PolicyPriorGMM
from algorithms.GPS.source.gps.algorithm.traj_opt.traj_opt import TrajOpt
from algorithms.GPS.source.gps.algorithm.policy_opt.policy_opt import PolicyOpt
from algorithms.GPS.source.gps.algorithm.policy_opt.lto_model import fully_connected_tf_network
from algorithms.GPS.source.gps.algorithm.policy.lin_gauss_init import init_cmaes_controller
from algorithms.GPS.source.gps.proto.gps_pb2 import CUR_LOC, PAST_OBJ_VAL_DELTAS, CUR_SIGMA, CUR_PS, PAST_LOC_DELTAS,PAST_SIGMA, ACTION
from algorithms.GPS.source.gps.algorithm.cost.cost_utils import RAMP_CONSTANT

try:
   import cPickle as pickle
except:
   import pickle
import copy



history_len = 40

TRAIN = True

input_dim = 10
num_inits_per_fcn = 1
init_locs = []
if TRAIN:
    num_fcns = 20
    train_fcns = range(num_fcns)
    test_fcns = range(num_fcns-1, num_fcns)
    fcn_ids = [12]
    fcn_names = ["BentCigar"]
    init_sigma_test = [1.28]
    #initialize the initial locations of the optimization trajectories
    init_locs.extend(list(np.random.randn(num_fcns-len(test_fcns), input_dim)))
    #initialize the initial sigma(step size) values
    init_sigmas = list(np.random.rand(num_fcns-len(test_fcns)))
    init_sigmas.extend(init_sigma_test)
    #append the initial locations of the conditions in the test set
    for i in test_fcns:
        init_locs.append([0]*input_dim)
    
else:
    num_fcns = 1
    # We don't do any training so we evaluate on all the conditions in the 'training set'
    train_fcns = range(num_fcns)
    test_fcns = train_fcns
    fcn_ids = [12]
    fcn_names = ["BentCigar"]
    init_sigmas = [1.28]*len(test_fcns)
    for i in test_fcns:
        init_locs.append([0]*input_dim)


cur_dir = os.path.dirname(os.path.abspath(__file__))


fcn_objs = []
fcns = []
for i in range(num_fcns//len(fcn_ids)):
    #instantiate BBOB functions based on their ID
    for i in fcn_ids:
        fcn_objs.append(bn.instantiate(i)[0])

for i,function in enumerate(fcn_objs):
    fcns.append({'fcn_obj': function, 'dim': input_dim, 'init_loc': list(init_locs[i]), 'init_sigma': init_sigmas[i]})

SENSOR_DIMS = {
    PAST_OBJ_VAL_DELTAS: history_len,
    CUR_PS: 1,
    CUR_SIGMA : 1,
    ACTION: 1,
    PAST_SIGMA: history_len
}

BASE_DIR = '/'.join(str.split(gps_filepath, '/')[:-2])
EXP_DIR = BASE_DIR + '/../examples/BentCigar' + '/'


common = {
    'experiment_name': 'CMA_' + \
            datetime.strftime(datetime.now(), '%m-%d-%y_%H-%M'),
    'experiment_dir': EXP_DIR,
    'data_files_dir': EXP_DIR + 'data_files/',
    'plot_filename': EXP_DIR + 'plot',
    'log_filename': EXP_DIR + 'log_data',
    'conditions': num_fcns,
    'train_conditions': train_fcns,
    'test_conditions': test_fcns,
    'test_functions': fcn_names
}

if not os.path.exists(common['data_files_dir']):
    os.makedirs(common['data_files_dir'])

agent = {
    'type': AgentCMAES,
    'world' : CMAESWorld,
    'init_sigma': 0.3,
    'popsize': 10,
    'n_min':10,
    'max_nfe': 200000,
    'substeps': 1,
    'conditions': common['conditions'],
    'dt': 0.05,
    'T': 50,
    'sensor_dims': SENSOR_DIMS,
    'state_include': [PAST_OBJ_VAL_DELTAS, CUR_SIGMA, CUR_PS, PAST_SIGMA],
    'obs_include': [PAST_OBJ_VAL_DELTAS, CUR_PS, PAST_SIGMA, CUR_SIGMA],
    'history_len': history_len,
    'fcns': fcns
}

algorithm = {
    'type': Algorithm,
    'conditions': common['conditions'],
    'train_conditions': train_fcns,
    'test_conditions': test_fcns,
    'test_functions': fcn_names,
    'iterations': 15,
    'inner_iterations': 4,
    'policy_dual_rate': 0.2,
    'init_pol_wt': 0.01,
    'ent_reg_schedule': 0.0,
    'fixed_lg_step': 3,
    'kl_step': 0.2,
    'min_step_mult': 0.01,
    'max_step_mult': 10.0,
    'sample_decrease_var': 0.05,
    'sample_increase_var': 0.1,
    'policy_sample_mode': 'replace',
    'exp_step_lower': 2,
    'exp_step_upper': 2
}

algorithm['init_traj_distr'] = {
    'type': init_cmaes_controller,
    'init_var': 0.01,
    'dt': agent['dt'],
    'T': agent['T']
}

algorithm['cost'] = {
    'type': Cost,
    'ramp_option': RAMP_CONSTANT,
    'wp_final_multiplier': 1.0,
    'weight': 1.0,
}

algorithm['dynamics'] = {
    'type': DynamicsLRPrior,
    'regularization': 1e-3,     # Increase this if Qtt is not PD during DGD
    'clipping_thresh': None,
    'prior': {
        'type': DynamicsPriorGMM,
        'max_clusters': 20,
        'min_samples_per_cluster': 20,
        'max_samples': 20,
        'strength': 1.0         # How much weight to give to prior relative to samples
    }
}

algorithm['traj_opt'] = {
    'type': TrajOpt,
}

algorithm['policy_opt'] = {
    'type': PolicyOpt,
    'network_model': fully_connected_tf_network,
    'iterations': 20000,
    'init_var': 0.01,
    'batch_size': 25,
    'solver_type': 'adam',
    'lr': 0.0001,
    'lr_policy': 'fixed',
    'momentum': 0.9,
    'weight_decay': 0.005,
    'use_gpu': 0,
    'weights_file_prefix': EXP_DIR + 'policy',
    'network_params': {
        'obs_include': agent['obs_include'],
        'sensor_dims': agent['sensor_dims'],
        'dim_hidden': [50, 50]
    }
}

algorithm['policy_prior'] = {
    'type': PolicyPriorGMM,
    'max_clusters': 20,
    'min_samples_per_cluster': 20,
    'max_samples': 20,
    'strength': 1.0,
    'clipping_thresh': None,
    'init_regularization': 1e-3,
    'subsequent_regularization': 1e-3
}

config = {
    'iterations': algorithm['iterations'],
    'num_samples': 25,
    'common': common,
    'agent': agent,
    'algorithm': algorithm,
    'train_conditions': train_fcns,
    'test_conditions': test_fcns,
    'test_functions': fcn_names,
    'policy_path':  EXP_DIR + 'data_files/policy_itr_14.pkl'
}

