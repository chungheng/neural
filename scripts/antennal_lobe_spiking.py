#!/usr/bin/env python
# coding: utf-8

import sys

import h5py
import matplotlib

import matplotlib
matplotlib.use('Agg')
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import copy
from scipy.optimize import curve_fit
import pycuda
import pycuda.autoinit
import pycuda.driver as drv
import pycuda.gpuarray as garray
from pycuda.elementwise import ElementwiseKernel
from tqdm import tqdm

from neural.utils import PSTH, compute_psth, generate_stimulus
from neural.recorder import CUDARecorder
from neural.network import Network
from neural.network.operator import Add, Repeat, Square, Sqrt, BlockSum, BlockMean
from neural.model import *
from neural import Model

import skcuda
import skcuda.misc
skcuda.misc.init()

sys.path.append('./src')
from model import OTP, OSNiLNSynapse, OSNAxonTerminalRate, NoisyConnorStevens, PoissonCSN, Mixture,    OSNAxonTerminal, iLNOSNSynapse, iLNOSNSynapseRate, PoissonCSN8,     PNDendTerminal, OSNPostLNSynapse, AlphaSynapse, Project
import scipy.io as sio


# ### Load affinity vector

# In[2]:


bd_data = np.load('../bionet-al/data/bd_peak_ss.npz')
aff = bd_data['bd'][11]

plt.figure()
plt.stem(np.arange(len(aff))+0.1, aff, 'r', use_line_collection=True);


# ### Setup simulation parameters

# In[3]:


dt = 1e-5
dtype = np.float64

# number of OSNs per receptor type
onum = 5
# number of receptors
rnum = len(aff)
# number of pre-LN
plnum = 100
# number of post-eLN
elnum = 50
# number of post-iLN
ilnum = 50

num = rnum*onum

aff_rep = np.repeat(aff, onum)

# project OSN spikes to OSN-preLN synapses
osn_preiln_proj = np.tile(np.arange(num), plnum)
# project preLN spikes to preLN-OSN synapses
preiln_osn_proj = np.tile(np.arange(plnum), rnum)


# ### Create the antennal lobe circuit
def _add(network, module, num, name, **kwargs):
    if not hasattr(module, 'Default_Params'):
        return network.add(module, num, name=name, **kwargs)
    params = copy.deepcopy(module.Default_Params)
    for key, val in kwargs.items():
        if np.isscalar(val):
            params[key] = np.full(num, val)
    for key, val in params.items():
        if np.isscalar(val):
            params[key] = np.full(num, val)
    return network.add(module, num, name=name, **params)
# In[4]:


nn = Network()

sti_1   = nn.input(name='waveform1',)
sti_2   = nn.input(name='waveform2',)
sti_mix = _add(nn, Mixture, num, name='Mixer', aff1=aff_rep, aff2=aff_rep[:])

# OSN
osn_otp = _add(nn, OTP, num=num, name='OSN OTP')
osn_bsg = _add(nn, NoisyConnorStevens, num=num, name='OSN BSG', sigma=0.0019/np.sqrt(dt))
# osn_bsg = _add(nn, PoissonCSN8, num=num)

# OSN to preLN synapse
# Project: Broadcast each OSN-BSG's spike to all 100 PreiLNs
osn_preiln = _add(nn, Project, plnum*num, name='OSN-preLN Spike Bus', index=osn_preiln_proj)
preiln_syn = _add(nn, OSNiLNSynapse, num=plnum*num, name='OSN-preLN Synapse', s=1e3, r=1./onum/80/80)

# preLN
preiln_agg = _add(nn, BlockSum, plnum*num, name='preLN Synapse Sum', block_size=num)
preiln_den = _add(nn, Sqrt, plnum, name='preLN Dendrite')

preiln_bsg = _add(nn, NoisyConnorStevens, num=num, name='preLN BSG', sigma=0.0019/np.sqrt(dt))
# preiln_bsg = _add(nn, PoissonCSN8, plnum)

# preLN to OSN synapse
preiln_osn = _add(nn, Project, plnum*rnum, name='preLN-OSN Spike Bus', index=preiln_osn_proj)
osn_presyn = _add(nn, iLNOSNSynapse,  plnum*rnum, name='preLN-OSN Synapse')
osn_presyn_agg = _add(nn, BlockSum, plnum*rnum, name='preLN-OSN Synapse Sum', block_size=plnum)
osn_presyn_rep = _add(nn, Repeat, rnum, name='preLN-OSN Synapse Repeat', rep_size=onum)

# OSN to PN Synapse
osn_axt = _add(nn, OSNAxonTerminal, num=num, name='OSN Axonal Terminal', s=1e3)
osn_pn_den = _add(nn, PNDendTerminal, num, name='OSN-PN Dendritic Terminal')
osn_pn_agg = _add(nn, BlockSum, num, name='OSN-PN Sum', block_size=onum)

# OSN to post-eLN Synapse
osn_eln_syn = _add(nn, OSNPostLNSynapse, num, name='OSN-eLN Synapse', p=1, a=35., g=0.1*50/onum)
eln_agg = _add(nn, BlockSum, num, name='OSN-eLN Synapse Sum', block_size=onum)

# OSN to post-iLN Synapse
osn_iln_syn = _add(nn, OSNPostLNSynapse, num, name='OSN-iLN Synapse', p=-1, C1 = 0.04, g=0.1*50/onum)
iln_agg = _add(nn, BlockSum, num, name='OSN-iLN Synapse Sum', block_size=onum)

# post-eLN and post-iLN
eln = _add(nn, NoisyConnorStevens, rnum, name='eLN BSG', sigma=0.)
iln = _add(nn, NoisyConnorStevens, rnum, name='iLN BSG', sigma=0.)
# eln = _add(nn, PoissonCSN8, rnum)
# iln = _add(nn, PoissonCSN8, rnum)

# post-LN to PN synaspe
eln_pn_syn = _add(nn, AlphaSynapse, rnum, name='eLN-PN Synapse', g = 0.4)
iln_pn_syn = _add(nn, AlphaSynapse, rnum, name='iLN-PN Synapse', g = -0.08)

# PN
pn_agg = _add(nn, Add, rnum, name='PN Dendritic Integration')
pn = _add(nn, NoisyConnorStevens, num=rnum, name='PN BSG', sigma=0.001/np.sqrt(dt))
# pn = _add(nn, PoissonCSN8, rnum)

# connect neurons and synapses
sti_mix(input1=sti_1, input2=sti_2)
osn_otp(stimulus=sti_mix.output)
osn_bsg(stimulus=osn_otp.I)

osn_preiln(input=osn_bsg.spike)
preiln_syn(stimulus=osn_preiln.output)
preiln_agg(input=preiln_syn.y)
preiln_den(input=preiln_agg.output)
preiln_bsg(stimulus=preiln_den.output)

preiln_osn(input=preiln_bsg.spike)

osn_presyn(stimulus=preiln_osn.output)
osn_presyn_agg(input=osn_presyn.y)
osn_presyn_rep(input=osn_presyn_agg.output)

osn_axt(stimulus=osn_bsg.spike, f=osn_presyn_rep.output)
osn_pn_den(stimulus=osn_axt.u)
osn_pn_agg(input=osn_pn_den.I)

osn_eln_syn(stimulus=osn_bsg.spike)
eln_agg(input=osn_eln_syn.I)
eln(stimulus=eln_agg.output)
eln_pn_syn(stimulus=eln.spike)

osn_iln_syn(stimulus=osn_bsg.spike)
iln_agg(input=osn_iln_syn.I)
iln(stimulus=iln_agg.output)
iln_pn_syn(stimulus=iln.spike)

pn_agg(input1=eln_pn_syn.I, input2=iln_pn_syn.I, input3=osn_pn_agg.output)
pn(stimulus=pn_agg.output)


nn.compile(dtype=dtype)

osn_pn_agg.record('output')
eln_pn_syn.record('I')
iln_pn_syn.record('I')
pn_agg.record('output')
pn.record('spike')

wav1 = generate_stimulus('step', dt, 5, (0.5, 4), 100)
wav2 = generate_stimulus('step', dt, 5, (0.15, 2.5), 0.)
t = np.arange(len(wav1))*dt

sti_1(wav1)
sti_2(wav2)
nn.run(dt, verbose=True)


# ### Visual simulation output
from cycler import cycler
cm = matplotlib.cm.hsv(np.linspace(0, 1, rnum))
custom_cycler = cycler(color=cm)

fig, axes = plt.subplots(1, 4, figsize=(15,4))

titles = [
    'iLN-PN Synaptic Current',
    'eLN-PN Synaptic Current',
    'OSN-PN Synaptic Current',
    'Total PN Synaptic Current'
]
for ax, title in zip(axes, titles):
    ax.grid()
    ax.set_prop_cycle(custom_cycler)
    ax.set_xlabel('Time, [s]', fontsize=14)
    ax.set_ylabel('current, [pA]', fontsize=14)
    ax.set_title(title)

for _x in iln_pn_syn.recorder.I:
    axes[0].plot(t,_x)

for _x in eln_pn_syn.recorder.I:
    axes[1].plot(t,_x)

for _x in osn_pn_agg.recorder.output:
    axes[2].plot(t,_x)

for _x in pn_agg.recorder.output:
    axes[3].plot(t,_x)

fig.tight_layout()
fig.savefig('fig/neural_output.pdf', dpi=200, bbox_inches='tight')

from cycler import cycler
cm = matplotlib.cm.hsv(np.linspace(0, 1, rnum))
custom_cycler = cycler(color=cm)

fig, ax = plt.subplots(1, 1, figsize=(15, 4))

ax.set_prop_cycle(custom_cycler)
ax.set_xlabel('Time, [s]', fontsize=14)
ax.set_ylabel('Channel Index', fontsize=14)
ax.set_title('PN Raster Plot', fontsize=16)

for i, _x in enumerate(pn.recorder.spike):
    spike_time = np.nonzero(_x)[0]*dt
    ax.plot(spike_time, np.ones(len(spike_time))*i, '|')
fig.savefig('fig/neural_spike_times.pdf', dpi=200, bbox_inches='tight')


# save all models
# for key, val in nn.containers.items():
#     obj = val.obj
#     cls = str(obj.__class__).split("'")[1]
#     if hasattr(obj, 'backend'):
#         module = cls.split('model.')[-1]
#         with open('./kernels/{}.cu'.format(module), 'w') as f:
#             f.writelines(obj.backend.src)
