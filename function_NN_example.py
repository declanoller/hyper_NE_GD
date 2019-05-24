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


nn.plotNetwork(show_plot=False, save_plot=True, fname=os.path.join('blog_output', '0.png'), plot_title=nn.get_analytic_NN_fn(latex_form=True))

nn.addConnectingWeight((0, 0, 3, 0))
nn.plotNetwork(show_plot=False, save_plot=True, fname=os.path.join('blog_output', '1.png'), plot_title=nn.get_analytic_NN_fn(latex_form=True))
print(nn.get_analytic_NN_fn())

nn.addAtomInBetween((0, 0, 3, 0))
nn.plotNetwork(show_plot=False, save_plot=True, fname=os.path.join('blog_output', '2.png'), plot_title=nn.get_analytic_NN_fn(latex_form=True))
print(nn.get_analytic_NN_fn())

nn.addConnectingWeight((1, 0, 4, 0))
nn.plotNetwork(show_plot=False, save_plot=True, fname=os.path.join('blog_output', '3.png'), plot_title=nn.get_analytic_NN_fn(latex_form=True))
print(nn.get_analytic_NN_fn())

nn.addConnectingWeight((2, 0, 4, 0))
nn.plotNetwork(show_plot=False, save_plot=True, fname=os.path.join('blog_output', '4.png'), plot_title=nn.get_analytic_NN_fn(latex_form=True))
print(nn.get_analytic_NN_fn())


#print('\n\nFunction: ', nn.get_analytic_NN_fn(latex_form=True))
nn.saveNetworkAsAtom(fname=os.path.join('blog_output', 'Atom_NAND.json'), atom_name='Atom_NAND')



#
