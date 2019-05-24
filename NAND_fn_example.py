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

show = False
save = True

nn.plotNetwork(show_plot=show, save_plot=save,
                fname=os.path.join('blog_output/NAND_fn_ex', 'NAND_atom_fn_0.png'),
                plot_title=nn.get_analytic_NN_fn(latex_form=True))

o = 0
w_tuple = (2*o, 0, N_in + 1 + o, 0)
nn.addConnectingWeight(w_tuple, std=3.0)

nn.plotNetwork(show_plot=show, save_plot=save,
                fname=os.path.join('blog_output/NAND_fn_ex', 'NAND_atom_fn_1.png'),
                plot_title=nn.get_analytic_NN_fn(latex_form=True))

nn.addAtomInBetween(w_tuple, atom_type='Atom_NAND')

nn.plotNetwork(show_plot=show, save_plot=save,
                fname=os.path.join('blog_output/NAND_fn_ex', 'NAND_atom_fn_2.png'),
                plot_title=nn.get_analytic_NN_fn(latex_form=True))

o = 0
w_tuple = (1 + 2*o, 0, N_in + 1 + N_out + o, 1)
nn.addConnectingWeight(w_tuple, std=3.0)

nn.plotNetwork(show_plot=show, save_plot=save,
                fname=os.path.join('blog_output/NAND_fn_ex', 'NAND_atom_fn_3.png'),
                plot_title=nn.get_analytic_NN_fn(latex_form=True))

#nn.plotNetwork(show_plot=True, save_plot=True, fname=os.path.join('blog_output', 'multi_NAND.png'))

#
