from agent_classes import *
from classes import Population
from BO_wrapper import bayes_opt, single_bay_opt

########### for grad free

const_args = {
    'N_evo_runs' : 1,
    'negate_runtime' : True,

    'N_pop' : 14,

    'weight_add_chance' : .2,
    'best_N_frac' : .2,

    'agent_class' : LogicAgentNand.LogicAgentNand,
    'weight_remove_chance' : 0.05,
    'weight_change_chance' : 0.98,
    'complex_atom_add_chance' : 0.0,
    'NN_output_type' : 'logic',

    'grad_desc' : False,
    'N_gen' : 256,
    'N_train_steps' : 300,
    'N_eval_steps' : 30,
    'N_trials_per_agent' : 1,
    'reward_stop_goal' : -0.05,
    'max_runtime' : 60
}

vary_args = {
    'atom_add_chance' : (.005, .5)
}



single_bay_opt(
    Population.multi_run_hist,
    const_args,
    vary_args,
    init_points=60,
    n_iter=60,
    alpha=1,
    kappa=5
)

exit()




const_args = {
    'N_evo_runs' : 1,
    'negate_runtime' : True,

    'N_pop' : 14,

    'weight_add_chance' : .6,
    'best_N_frac' : .4,

    'agent_class' : LogicAgentNand.LogicAgentNand,
    'weight_remove_chance' : 0.1,
    'weight_change_chance' : 0.0,
    'complex_atom_add_chance' : 0.0,
    'NN_output_type' : 'logic',

    'grad_desc' : True,
    'N_gen' : 80,
    'N_train_steps' : 300,
    'N_eval_steps' : 30,
    'N_trials_per_agent' : 1,
    'reward_stop_goal' : -0.05,
    'max_runtime' : 60
}

vary_args = {
    'atom_add_chance' : (.1, .95)
}



single_bay_opt(
    Population.multi_run_hist,
    const_args,
    vary_args,
    init_points=60,
    n_iter=60,
    alpha=1,
    kappa=5
)




single_bay_opt(
    Population.multi_run_hist,
    const_args,
    vary_args,
    init_points=60,
    n_iter=60,
    alpha=5,
    kappa=5
)

#################### Did these two



single_bay_opt(
    Population.multi_run_hist,
    const_args,
    vary_args,
    init_points=60,
    n_iter=60,
    alpha=5,
    kappa=10
)





single_bay_opt(
    Population.multi_run_hist,
    const_args,
    vary_args,
    init_points=60,
    n_iter=60,
    alpha=1,
    kappa=10
)




exit()












const_args = {
    'N_evo_runs' : 20,
    'negate_runtime' : True,

    'agent_class' : LogicAgentNand.LogicAgentNand,
    'weight_remove_chance' : 0.1,
    'weight_change_chance' : 0.0,
    'complex_atom_add_chance' : 0.0,
    'NN_output_type' : 'logic',

    'grad_desc' : True,
    'N_gen' : 80,
    'N_train_steps' : 300,
    'N_eval_steps' : 30,
    'N_trials_per_agent' : 1,
    'reward_stop_goal' : -0.05,
    'max_runtime' : 60
}

vary_args = {
    'N_pop' : (4, 32),
    'atom_add_chance' : (.05, .9),
    'weight_add_chance' : (.2, .9),
    'best_N_frac' : (.05, .5)
}



bayes_opt(
    Population.multi_run_hist,
    const_args,
    vary_args,
    n_iter=3
)

exit()









######### For grad desc method
### these settings worked pretty great:
Population.multi_run_hist(
N_evo_runs=10,
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

exit()





######### For grad free method

Population.multi_run_hist(40, fname='noGD_nand',
agent_class=LogicAgentNand.LogicAgentNand,
N_pop=16,
atom_add_chance=0.05,
weight_add_chance=0.2,
weight_remove_chance=0.05,
weight_change_chance=0.98,
complex_atom_add_chance=0.0,
NN_output_type='logic',
best_N_frac = 0.2,

grad_desc = False,
N_gen=256,
N_train_steps = 300,
N_eval_steps = 30,
N_trials_per_agent = 1,
reward_stop_goal = -0.05,
max_runtime = 90

)

exit()














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
