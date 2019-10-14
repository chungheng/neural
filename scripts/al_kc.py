#!/usr/bin/env python
# coding: utf-8

# In[5]:


# official libraries
import sys
from itertools import product, chain

# third-party libraries
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt

import numpy as np

import pycuda
import pycuda.autoinit

import skcuda
import skcuda.misc
skcuda.misc.init()

from tqdm import tqdm

#
from neural import Model
from neural.utils import generate_stimulus
from neural.network import Network
from neural.network.operator import Repeat, Sqrt, BlockSum, Dot, Add, BlockMean


from model import (OTP, OSNiLNSynapse, OSNAxonTerminalRate,
                   PostLN, KCDend, ReLU, NoisyConnorStevens,
                   PoissonCSN, Mixture)

BACKEND = 'cuda'
# ## Load Affinity Data

# In[6]:


bd_data = np.load('./bd_peak_ss.npz')['bd']
aff_1 = bd_data[11]
aff_2 = bd_data[41]
plt.figure()
plt.stem(np.arange(len(aff_1))+0.1, aff_1, 'r')
plt.stem(np.arange(len(aff_2))+0.6, aff_2, 'b')
aff = (bd_data.T/np.linalg.norm(bd_data, axis=1)).T


# ## Specify Parameters

# In[7]:


dt = 1e-4
dtype = np.float64

rnum = len(aff_1)
knum = 1000

np.random.seed(10)
idx = np.arange(rnum)
pn_kc_mat = np.zeros((knum, rnum))
for i in range(knum):
    np.random.shuffle(idx)
    pn_kc_mat[i][idx[:6]] = 1
iln_mat = np.eye(rnum, dtype=dtype)
eln_mat = np.eye(rnum, dtype=dtype)


print('[AL_KC] Constructing Network...', flush=True)
nn = Network()#solver='euler')#'scipy_ivp:RK45')

# odorants and binding vector
sti_1   = nn.input(name='waveform1',)
sti_2   = nn.input(name='waveform2',)
sti_mix = nn.add(Mixture, rnum, name='Mixer', aff1=aff_1, aff2=aff_2)

# osn
osn_otp = nn.add(OTP, num=rnum, name='OTP')
osn_bsg = nn.add(PoissonCSN, name='OSN-BSG', num=rnum)
osn_axt = nn.add(OSNAxonTerminalRate, name='OSN-AT', num=rnum)

# osn to pre-synaptic LN pathway, i.e., predictive coding circuit
osn2pln = nn.add(OSNiLNSynapse, name='OSN-pLN', num=rnum)
pln_den = nn.add(BlockSum, rnum, name='pLN-DT', block_size=rnum)
pln     = nn.add(Sqrt, 1, name='pLN')
pln2osn = nn.add(Repeat, 1, name='pLN-OSN', rep_size=rnum)

# osn to post-synaptic LN pathway, i.e., on-off circuit
osn2iln = nn.add(Dot, name='OSN-iLN', multiplier=iln_mat, num=rnum)
osn2eln = nn.add(Dot, name='OSN-eLN', multiplier=eln_mat, num=rnum)

iln = nn.add(PostLN, name='iLN', num=rnum, p=-1, s=100., a=-1.2e-1)
eln = nn.add(PostLN, name='eLN', num=rnum, p=1, s=100., a=5e-2)

# pn
pn_den = nn.add(Add, rnum, name='PN-DT')
pn     = nn.add(ReLU, rnum, name='PN')

# kc
pn2kc  = nn.add(Dot, name='PN-KC', num=knum, multiplier=pn_kc_mat)
kc_den = nn.add(KCDend, name='KC-DT', num=knum)
kc     = nn.add(ReLU, name='KC', num=knum, th=2.0)

# apl
kc2apl = nn.add(BlockMean, knum, name='KC-APL', block_size=knum)
apl2kc = nn.add(Repeat, 1, name='APL-KC', rep_size=knum)

sti_mix(input1=sti_1, input2=sti_2)

osn_otp(stimulus=sti_mix.output)
osn_bsg(I=osn_otp.I)

osn2pln(stimulus=osn_bsg.x)
pln_den(input=osn2pln.y)
pln(input=pln_den.output)
pln2osn(input=pln.output)

osn_axt(stimulus=osn_bsg.x, f=pln2osn.output)

osn2iln(input=osn_bsg.x)
osn2eln(input=osn_bsg.x)

iln(stimulus=osn2iln.output)
eln(stimulus=osn2eln.output)

pn_den(input1=osn_axt.u, input2=iln.z, input3=eln.z)
pn(stimulus=pn_den.output)

pn2kc(input=pn.x)
kc_den(stimulus=pn2kc.output, f=apl2kc.output)
kc(stimulus=kc_den.x)

kc2apl(input=kc.x)
apl2kc(input=kc2apl.output)


nn.compile(dtype=dtype)

# specify attributes/fields to record
# sti_mix.record('output')
osn_otp.record('uh', 'I')
# osn_bsg.record('x', 'cx')
# pln2osn.record('output')
osn_axt.record('x', 'u')
# osn2pln.record('y')
# pln.record('output')
# osn2iln.record('output')
# osn2eln.record('output')
# iln.record('z')
# eln.record('z')
pn.record('x')
# kc_den.record('x', 'z')
kc.record('x')


# ## Generate Stimuli
wav1 = [
    generate_stimulus('step', dt, 6.5, (0.5, 6.5), 20),
    generate_stimulus('step', dt, 6.5, (0.5, 6.5), 20),
    generate_stimulus('step', dt, 6.5, (0.5, 6.5), 20),
    generate_stimulus('step', dt, 6.5, (0.5, 6.5), 20),
    generate_stimulus('step', dt, 6.5, (0.5, 6.5), 20)
]
wav2 = [
    generate_stimulus('step', dt, 6.5, (0.5, 4.5), 20),
    generate_stimulus('step', dt, 6.5, (2.5, 4.5), 100),
    generate_stimulus('step', dt, 6.5, (2.5, 4.5), 120),
    generate_stimulus('ramp', dt, 6.5, (2.5, 4.5), 120, ratio=0.6),
    generate_stimulus('parabola', dt, 6.5, (2.5, 4.5), 130, ratio=0.9)
]
t = np.arange(len(wav1[0]))*dt
idx = np.logical_and(t>3, t<4)
wav2[2][idx] += np.random.randn(sum(idx))*10

## Run Simulation
result = {}
for i, (_w1, _w2) in enumerate(zip(wav1, wav2)):
    sti_1(_w1)
    sti_2(_w2)
    nn.run(dt, verbose=True)

    result[i] = {
        'ky': np.copy(kc.recorder.x),
        'u': np.copy(osn_axt.recorder.u),
        'p': np.copy(pn.recorder.x),
    }


# ### Plot Result

# In[19]:


w_max = np.max([np.max(wav1), np.max(wav2)])

fig, axes = plt.subplots(4, 7, figsize=(20,15),
    gridspec_kw={
        'width_ratios': [1, 1, 1, 1, 1, 0.05, 0.2],
        'height_ratios': [1, 1, 3, 1]})

for i, (_w1, _w2) in enumerate(zip(wav1, wav2)):
    axes[0,i].plot(t, _w1, '-', color='k', label='Odorant 1')
    axes[0,i].plot(t, _w2, '-.', color=[0.4, 0.4, 0.4], label='Odorant 2')
    axes[0,i].legend(loc=2)
    res = result[i]['p']

    axes[1,i].imshow(res,
        cmap=matplotlib.cm.viridis,
        aspect='auto',
        vmin=0,
        vmax=1.)

    res = result[i]['ky']
    axes[2,i].imshow(res,
        cmap=matplotlib.cm.inferno,
        aspect='auto')

    axes[3,i].plot(t, 100*np.mean(res>0, axis=0))

for ax in axes[0][:-2]:
    ax.set_xlim([t[0], t[-1]])
    ax.set_ylim([0, w_max])
    ax.grid(True)
for ax in axes[:3].ravel():
    ax.set_xticklabels([])

xtick = np.arange(0, len(t), 5000)

for ax in axes[-1,:-2]:
    ax.grid()
    ax.set_ylim([0, 20.])
    ax.set_xlim([t[0], t[-1]])
    ax.set_xlabel('Time, [sec]', fontsize=16)

for ax in axes[:, 1:-2].ravel():
    ax.set_yticklabels([])
    ax.tick_params('both', length=0, width=0, which='major')

for ax in axes[1, :-2].ravel():
    ax.set_yticks(np.arange(0, rnum, 1)-0.5, minor=True)
    ax.tick_params('both', length=0, width=0, which='minor')
    # Gridlines based on minor ticks
    ax.grid(which='minor', color='w', linestyle='-', linewidth=0.15)


axes[0,0].set_ylabel('Concentration, [ppm]', fontsize=16)
axes[1,0].set_ylabel('PN Index', fontsize=16)
axes[2,0].set_ylabel('KC Index', fontsize=16)
axes[3,0].set_ylabel('Active KC, [%]', fontsize=16)

cb1 = matplotlib.colorbar.ColorbarBase(
    axes[0,-2],
    cmap=matplotlib.cm.gnuplot,
    norm= matplotlib.colors.Normalize(vmin=0, vmax=1))
axes[0,-2].yaxis.set_label_position("left")
axes[0,-2].yaxis.tick_right()
cb1.set_label('Affinity Value')

cb2 = matplotlib.colorbar.ColorbarBase(
    axes[1,-2],
    cmap=matplotlib.cm.viridis,
    norm=matplotlib.colors.Normalize(vmin=0, vmax=1.))
axes[1,-2].yaxis.set_label_position("left")
axes[1,-2].yaxis.tick_right()
cb2.set_label('PN Activity')

cb3 = matplotlib.colorbar.ColorbarBase(
    axes[2,-2],
    cmap=matplotlib.cm.inferno,
    norm=matplotlib.colors.Normalize(vmin=0, vmax=1.))
axes[2,-2].yaxis.set_label_position("left")
axes[2,-2].yaxis.tick_right()
cb3.set_label('KC Activity')

# plot affinity matrix
aff_com = np.vstack([aff_1, aff_2])
aff_com = aff_com / np.reshape(np.linalg.norm(aff_com, axis=1), (-1,1))

axes[0][-1].imshow(
    aff_com.T,
    cmap=matplotlib.cm.gnuplot,
    interpolation='none', vmin=0, vmax=1, aspect='auto');

axes[0][-1].set_xticks(np.arange(0, aff_com.shape[0]))
axes[0][-1].yaxis.tick_right()
axes[0][-1].yaxis.set_label_position("right")
axes[0][-1].set_yticks(np.arange(0, rnum, 2))

# Minor ticks
axes[0][-1].set_xticks(np.arange(0, 3, 1)-0.5, minor=True)
axes[0][-1].set_yticks(np.arange(0, len(aff_2), 1)-0.5, minor=True)

# Gridlines based on minor ticks
axes[0][-1].grid(which='minor', color='w', linestyle='-', linewidth=0.5)
axes[0][-1].tick_params('both', length=0, width=0, which='minor')

axes[0][-1].set_ylabel('Channel Index')
plt.tight_layout()

axes[1][-1].remove()
axes[2][-1].remove()
axes[3][-2].remove()
axes[3][-1].remove()

axes[0][-1].set_xticklabels(
    ['Odorant: {}'.format(x) for x in [1, 2, 3]],
    rotation='vertical')

fig.savefig('./figures/al_kc_{}.pdf'.format(BACKEND), dpi=200, bbox_inches='tight')

