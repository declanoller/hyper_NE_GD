from agent_classes import *
from classes import Atom, EvoAgent, HyperEPANN, Population


#################################### GD method

run_notes = 'GD, MUX, 0.2 atom add, 0.8 weight add, 0.95 complex add'

Population.multi_run_hist(
notes = run_notes,
N_evo_runs=30,
fname='GD_MUX',
agent_class=LogicAgentMux.LogicAgentMux,
N_pop=16,
atom_add_chance=0.5,
weight_add_chance=0.8,
weight_remove_chance=0.0,
weight_change_chance=0.0,
complex_atom_add_chance=0.95,
NN_output_type='logic',
best_N_frac = 0.4,

grad_desc = True,
reset_weights = False,
N_batch=1,


N_gen=128,
N_train_steps = 300,
N_eval_steps = 30,
N_trials_per_agent = 1,
reward_stop_goal = -0.02,
max_runtime = 240

)




exit()








######### For grad desc method
### found via BO:
Population.multi_run_hist(
N_evo_runs=40,
fname='GD_nand',
agent_class=LogicAgentNand.LogicAgentNand,
N_pop=14,
atom_add_chance=0.86,
weight_add_chance=0.62,
weight_remove_chance=0.1,
weight_change_chance=0.0,
complex_atom_add_chance=0.0,
NN_output_type='logic',
best_N_frac = 0.38,

grad_desc = True,
N_gen=80,
N_train_steps = 300,
N_eval_steps = 30,
N_trials_per_agent = 1,
reward_stop_goal = -0.05,
max_runtime = 60

)

exit()




### these settings worked pretty great:
Population.multi_run_hist(40, fname='GD_nand',
agent_class=LogicAgentNand.LogicAgentNand,
N_pop=16,
atom_add_chance=0.5,
weight_add_chance=0.8,
weight_remove_chance=0.1,
weight_change_chance=0.0,
complex_atom_add_chance=0.0,
NN_output_type='logic',
best_N_frac = 0.4,

grad_desc = True,
N_gen=80,
N_train_steps = 300,
N_eval_steps = 30,
N_trials_per_agent = 1,
reward_stop_goal = -0.05,
max_runtime = 90

)

















runtime_dist = []

for i in range(1):

    p1 = Population.Population(
    agent_class=LogicAgentHalfAdder.LogicAgentHalfAdder,
    N_pop=16,
    atom_add_chance=0.1,
    weight_add_chance=0.1,
    weight_remove_chance=0.0,
    weight_change_chance=0.8,
    complex_atom_add_chance=0.0,
    NN_output_type='logic',
    best_N_frac = 0.2
    )

    '''p1 = Population.Population(
    agent_class=LogicAgentHalfAdder.LogicAgentHalfAdder,
    N_pop=4,
    atom_add_chance=0.2,
    weight_add_chance=0.8,
    weight_remove_chance=0.0,
    weight_change_chance=0.0,
    complex_atom_add_chance=0.0,
    NN_output_type='logic',
    best_N_frac = 0.2
    )'''

    d = p1.evolve(
        grad_desc = True,
        N_gen=256,
        N_train_steps = 300,
        N_eval_steps = 30,
        N_trials_per_agent = 1,
        reward_stop_goal = -0.05)

    runtime_dist.append(d['runtime'])


plt.hist(runtime_dist)
plt.xlabel('Run times (s)')


exit()


p1 = Population.Population(
agent_class=LogicAgentFullAdder.LogicAgentFullAdder,
N_pop=32,
atom_add_chance=0.5,
weight_add_chance=0.8,
weight_remove_chance=0.0,
weight_change_chance=0.0,
complex_atom_add_chance=0.7,
NN_output_type='logic',
best_N_frac = 0.5
)

p1.evolve(
N_gen=32,
N_episode_steps=500,
N_eval_steps=50,
N_trials_per_agent=1,
reward_stop_goal = -0.05)

#, N_runs_with_best=2, record_final_runs=False, show_final_runs=False
best_individ = p1.population[0]

print('0 0:', best_individ.forwardPass([0,0]))
print('0 1:', best_individ.forwardPass([0,1]))
print('1 0:', best_individ.forwardPass([1,0]))
print('1 1:', best_individ.forwardPass([1,1]))
#best_individ.NN.saveNetworkAsAtom()

#best_individ.plotNetwork()

exit()








#
