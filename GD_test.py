from agent_classes import *
from classes import Atom, EvoAgent, HyperEPANN, Population
import matplotlib.pyplot as plt
import numpy as np


e = EvoAgent.EvoAgent(
    agent_class=LogicAgentNand.LogicAgentNand,
    NN_output_type='logic'
)


e.NN.addConnectingWeight((0, 0, 3, 0))
e.NN.addAtomInBetween((0, 0, 3, 0))
e.NN.addConnectingWeight((1, 0, 4, 0))
e.NN.addConnectingWeight((2, 0, 4, 0))


################# For testing lr

train_curves = []

plt.figure(figsize=(14,8))

N_runs = 20

for i,lr in enumerate([10**0, 10**-1, 10**-2, 10**-3]):
    print('Testing LR ', lr)
    t_c = []

    for r in range(N_runs):
        print('Run ', r)
        ret = e.runEpisode(
            grad_desc = True,
            reset_weights = True,
            N_train_steps = 2000,
            N_eval_steps = 30,
            lr=lr,
            N_batch = 1
        )

        t_c.append(np.array(ret['train_curve']))

    m = np.mean(t_c, axis=0)
    sd = np.std(t_c, axis=0)

    plt.subplot(2,2,i+1)
    plt.fill_between(np.array(range(len(m))), m - sd, m + sd, facecolor='lightcoral', alpha=0.5)
    plt.plot(np.array(range(len(m))), m, color='firebrick', label=f'LR = {lr}')
    plt.legend()


plt.savefig('learning_rates_mean.png')
plt.show()


'''

################# For testing batch size

train_curves = []

plt.figure(figsize=(14,10))

for b in [1,2,4,8]:

    ret = e.runEpisode(
        grad_desc = True,
        reset_weights = True,
        N_train_steps = 300,
        N_eval_steps = 30,
        N_batch = b
    )

    plt.plot(ret['train_iters'], np.array(ret['train_curve'])/b, label=f'Batch size = {b}')

plt.legend()
plt.savefig('batch_sizes.png')
plt.show()

'''


#
