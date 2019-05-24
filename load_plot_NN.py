from agent_classes import *
from classes import Atom, EvoAgent, HyperEPANN, Population
import matplotlib.pyplot as plt
import numpy as np
import os

N_out = 1
N_in = 2*N_out

nn = HyperEPANN.HyperEPANN(
    N_inputs = N_in,
    N_outputs = N_out
)

f = os.path.join('/home/declan/Documents/code/hyper_NE_GD/output/multi_run_output/multi_24-05-2019_07-55-38/evolve_LogicAgentMux_24-05-2019_07-55-38',
                'bestNN_LogicAgentMux_24-05-2019_07-55-38.json')
nn.loadNetworkFromFile(fname=f)


nn.plotNetwork(show_plot=True, save_plot=False, fname=os.path.join('blog_output', '0.png'))


#
