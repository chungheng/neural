#!/usr/bin/env python
# coding: utf-8

# # Demonstration of Basic Neuron Models

# In[1]:


import numpy as np
import scipy.io as sio
import matplotlib
import matplotlib.pyplot as plt
import sys

from neural.model.neuron import *
from neural.plot import plot_multiple
from neural.utils import generate_stimulus

from scipy.integrate import solve_ivp
from tqdm import tqdm

# Define input stimulus.

# In[3]:


dt  = 1e-4
dur = 0.2

waveform = generate_stimulus('step', dt, dur-dt/2, (0.05, 0.15), 20.)
t = np.arange(0, len(waveform)*dt-dt/2, dt)

# Simulate neuron models.

# In[6]:


model_list = [IAF, LeakyIAF, HodgkinHuxley, Wilson, Rinzel, ConnorStevens]
record = {key:np.zeros(len(waveform)) for key in model_list}

for M in model_list:
    model = M(solver='euler')
    for i, wav in tqdm(enumerate(waveform)):
        model.update(dt, stimulus=wav)
        record[M][i] = model.v


# Plot simulation result.

# In[7]:


# fig, axes = plot_multiple(
#     t,
#     (record[IAF], {'color':'c', 'label':'IAF'}),
#     (record[LeakyIAF], {'color':'b', 'label':'Leaky AF'}),
#     (record[HodgkinHuxley], {'color':'r', 'label':'Hodgkin-Huxley'}),
#     (record[Wilson], {'color':'g', 'label':'Wilson'}),
#     (record[Rinzel], {'color':'purple', 'label':'Rinzel'}),
#     (record[ConnorStevens], {'color':'m', 'label':'Connor-Stevens'}),
#     xlim=(0,dur), figw=8, figh=3, ylabel='conductance')
