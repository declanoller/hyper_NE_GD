from agent_classes import *
from classes import Atom, EvoAgent, HyperEPANN, Population
import matplotlib.pyplot as plt
import numpy as np
import os


N_out = 8
N_in = 2*N_out

nn = HyperEPANN.HyperEPANN(
    N_inputs = N_in,
    N_outputs = N_out
)

for o in range(N_out):
    w_tuple = (2*o, 0, N_in + 1 + o, 0)
    nn.addConnectingWeight(w_tuple)
    nn.addAtomInBetween(w_tuple, atom_type='Atom_NAND')

for o in range(N_out):
    w_tuple = (1 + 2*o, 0, N_in + 1 + N_out + o, 1)
    nn.addConnectingWeight(w_tuple)

nn.plotNetwork(show_plot=True, save_plot=True, fname=os.path.join('blog_output', 'multi_NAND.png'))

#
