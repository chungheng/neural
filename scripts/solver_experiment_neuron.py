#!/usr/bin/env python
# coding: utf-8
import numpy as np
import scipy.io as sio
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys

import neural
from neural.model.neuron import *
from neural.plot import plot_multiple
from neural.utils import generate_stimulus

from sklearn.metrics import pairwise_distances
from scipy.integrate import solve_ivp

from tqdm import tqdm
import time

SCIPY_SOLVERS = ['RK45', 'RK23', 'Radau', 'BDF', 'LSODA']
NEURAL_SOLVERS = [k for k in list(neural.Model.solver_alias.keys()) if k != 'scipy_ivp']

dur = 0.2
config = {'dt': [1e-5, 5e-5, 1e-4, 5e-4],
          'solvers':  {'scipy':SCIPY_SOLVERS, 'neural':NEURAL_SOLVERS}}

model_list = [IAF, LeakyIAF, HodgkinHuxley, Wilson, Rinzel, ConnorStevens]
record = {'scipy':{}, 'neural':{}}
time_taken = {'scipy':{}, 'neural':{}}

for program in ['neural', 'scipy']:
    program_solvers = config['solvers'][program]
    for solver in  program_solvers:
        record[program][solver] = {}
        time_taken[program][solver] = {}
        for dt in config['dt']:
            print("{} - {}: dt= {} ...".format(program, solver, dt), flush=True)
            waveform = generate_stimulus('step', dt, dur-dt/2, (0.05, 0.15), 20.)
            time_taken[program][solver][dt] = {key:None for key in model_list}
            record[program][solver][dt] = {key:np.zeros(len(waveform)) for key in model_list}
            t = np.arange(0, len(waveform)*dt-dt/2, dt)
            for M in model_list:
                if program == 'scipy':
                    model = M(solver='scipy_ivp')
                    _update = lambda _dt, _stim: model.update(_dt, stimulus=_stim, solver=solver)
                else:
                    model = M(solver=solver)
                    _update = lambda _dt, _stim: model.update(_dt, stimulus=_stim)
                t0 = time.time()
                for i, wav in enumerate(waveform):
                    try: 
                        _update(dt, wav)
                        record[program][solver][dt][M][i] = model.v
                    except Exception as e:
                        record[program][solver][dt][M][i] = None
                        print('\t [{}-{}] !!! {} Errored, error {}'.format(program, solver, M, e))
                        break
                t1 = time.time()
                time_taken[program][solver][dt][M] = t1-t0
                print('\t {}, Time ={:.2f}'.format(M, t1-t0))

fig,axes = plt.subplots(len(model_list), len(config['dt'])*3, figsize=(25, 20),
                        gridspec_kw={'width_ratios':[1, 1, 0.5]*len(config['dt'])})
cmaps= {'scipy': plt.get_cmap('winter', len(SCIPY_SOLVERS)),
        'neural': plt.get_cmap('summer', len(NEURAL_SOLVERS))}
for dt_idx, dt in enumerate(config['dt']):
    for model_idx, model in enumerate(model_list):
        stacked_dists = []
        stacked_methods = []
        for program_idx, (program, program_solvers) in enumerate(config['solvers'].items()):
            _ax = axes[model_idx, 3*dt_idx + program_idx]
            _ax_mat = axes[model_idx, 3*dt_idx+2]
            if model_idx == 0:
                _ax.set_title('dt = {:.1e}'.format(dt))

            _model_name = str(model).split('neuron.')[-1].split("'")[0]
            _ax.set_ylabel(_model_name)

            for solver_idx, solver in enumerate(program_solvers):
                color = cmaps[program](solver_idx)
                _ax.plot(np.arange(len(record[program][solver][dt][model]))*dt,
                         record[program][solver][dt][model],
                         color=color, label='{} - {}'.format(program, solver))
                stacked_dists.append(record[program][solver][dt][model])
                stacked_methods.append('{}-{}'.format(program.upper()[0], solver))
            _ax.legend(fontsize=2)
        stacked_dists = np.vstack([k[np.newaxis, :]  for k in stacked_dists])
        stacked_dists[np.isnan(stacked_dists)] = 1e9
        stacked_dists[stacked_dists>1e5] = 1e5
        p_dists = pairwise_distances(stacked_dists, metric='euclidean')
        df = pd.DataFrame(p_dists)
        df.columns = stacked_methods
        df.index = stacked_methods
        df[df.isna()] = -1
        sns.heatmap(df, annot=False, ax=_ax_mat, annot_kws={'fontsize': 4})

fig.tight_layout()
fig.savefig('./figures/compare_methods.pdf', dpi=200, bbox_inches='tight')


fig,axes = plt.subplots(len(model_list), len(config['dt']), figsize=(25, 20))
cmaps= {'scipy': plt.get_cmap('winter', len(SCIPY_SOLVERS)),
        'neural': plt.get_cmap('summer', len(NEURAL_SOLVERS))}
for dt_idx, dt in enumerate(config['dt']):
    for model_idx, model in enumerate(model_list):
        stacked_dists = []
        stacked_methods = []
        _ax_mat = axes[model_idx, dt_idx]
        if model_idx == 0:
            _ax_mat.set_title('dt = {:.1e}'.format(dt))
        for program_idx, (program, program_solvers) in enumerate(config['solvers'].items()):
            _model_name = str(model).split('neuron.')[-1].split("'")[0]
            for solver_idx, solver in enumerate(program_solvers):
                color = cmaps[program](solver_idx)
                stacked_dists.append(record[program][solver][dt][model])
                stacked_methods.append('{}-{}'.format(program.upper()[0], solver))
        stacked_dists = np.vstack([k[np.newaxis, :]  for k in stacked_dists])
        stacked_dists[np.isnan(stacked_dists)] = 1e9
        stacked_dists[stacked_dists>1e5] = 1e5
        p_dists = pairwise_distances(stacked_dists, metric='euclidean')
        df = pd.DataFrame(p_dists)
        df.columns = stacked_methods
        df.index = stacked_methods
        df[df.isna()] = -1
        sns.heatmap(df, annot=False, ax=_ax_mat, annot_kws={'fontsize': 4})

fig.tight_layout()
fig.savefig('./figures/compare_methods_mat.pdf', dpi=200, bbox_inches='tight')


fig,axes = plt.subplots(len(model_list), len(config['dt'])*2, figsize=(25, 20),
                        gridspec_kw={'width_ratios':[1, 1]*len(config['dt'])})
cmaps= {'scipy': plt.get_cmap('Set1', len(SCIPY_SOLVERS)),
        'neural': plt.get_cmap('Set1', len(NEURAL_SOLVERS))}
for dt_idx, dt in enumerate(config['dt']):
    for model_idx, model in enumerate(model_list):
        stacked_dists = []
        stacked_methods = []
        for program_idx, (program, program_solvers) in enumerate(config['solvers'].items()):
            _ax = axes[model_idx, 2*dt_idx + program_idx]
            if model_idx == 0:
                _ax.set_title('dt = {:.1e}'.format(dt))

            _model_name = str(model).split('neuron.')[-1].split("'")[0]
            _ax.set_ylabel(_model_name)

            for solver_idx, solver in enumerate(program_solvers):
                color = cmaps[program](solver_idx)
                _ax.plot(np.arange(len(record[program][solver][dt][model]))*dt,
                         record[program][solver][dt][model],
                         color=color, label='{} - {}'.format(program, solver),
                         linewidth=0.2)
                stacked_dists.append(record[program][solver][dt][model])
                stacked_methods.append('{}-{}'.format(program.upper()[0], solver))
            _ax.legend(fontsize=5)
fig.tight_layout()
fig.savefig('./figures/compare_methods_traces.pdf', dpi=200, bbox_inches='tight')

# Plot Time Taken
fig,axes = plt.subplots(1, len(config['dt']), figsize=(20, 4))
cmaps= {'scipy': plt.get_cmap('winter', len(SCIPY_SOLVERS)),
        'neural': plt.get_cmap('summer', len(NEURAL_SOLVERS))}
for dt_idx, dt in enumerate(config['dt']):
    _ax = axes[dt_idx]
    _ax.set_title('dt = {:.1e}'.format(dt))
    stacked_times = {}
    for program_idx, (program, program_solvers) in enumerate(config['solvers'].items()):
        for solver in program_solvers:
            _curr_times = [] # time taken for all models of a given solver
            _curr_method = '{}-{}'.format(program.upper()[0], solver)
            model_names = []
            for model_idx, model in enumerate(model_list):
                _model_name = str(model).split('neuron.')[-1].split("'")[0]
                _curr_times.append(time_taken[program][solver][dt][model])
                model_names.append(_model_name)
            stacked_times[_curr_method] = _curr_times
    df = pd.DataFrame(stacked_times, index=model_names)
    sns.heatmap(df, annot=True, ax=_ax, annot_kws={'fontsize': 4})
fig.tight_layout()
fig.savefig('./figures/compare_methods_time.pdf', dpi=200, bbox_inches='tight')

np.save('compare_methods', {'record':record, 'time':time_taken})
