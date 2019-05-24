
import torch
from classes import *
from agent_classes import *
import sympy

#sym1 = [sympy.symbols()]


#exit()





ea = EvoAgent.EvoAgent(
agent_class=LogicAgentNand.LogicAgentNand,
NN_output_type='logic',
atom_add_chance=0.15,
weight_add_chance=0.5
)

[ea.mutate() for _ in range(35)]

print(ea.NN.NN_fn)

ea.plotNetwork(save_plot=True)

ea.runEpisode(10000, 100)

ea.plotNetwork(save_plot=True)


print('0 0:', ea.forwardPass([0,0]))
print('0 1:', ea.forwardPass([0,1]))
print('1 0:', ea.forwardPass([1,0]))
print('1 1:', ea.forwardPass([1,1]))


exit()











nn = HyperEPANN.HyperEPANN(verbose=True)
nn.mutateAddWeight(1)
nn.mutateAddWeight(1)

print(nn.atom_weight_tensors)

nn_in = [5.0,8.0]
out1 = nn.forwardPass(nn_in)
print(out1)
print(out1.shape)
t = torch.tensor([.5, .8])
#t = [.5, .8]
print(t)
print(t.shape)
nn.backProp(out1, t)
print(nn.atom_weight_tensors)

out1 = nn.forwardPass(nn_in)
print(out1)
print(out1.shape)


exit()








print(nn.analytic_NN_fn)
print(nn.NN_fn)

#print(nn.NN_fn([5,8]))

print(nn.forwardPass([5,8]))

print(nn.atom_weight_tensors[0].grad)

nn.plotNetwork()









import sympy

import torch


#f = 'w1*i1 + w2*i2 + b'
i1 = sympy.symbols('i1')
i2 = sympy.symbols('i2')
w1 = sympy.symbols('w1')
w2 = sympy.symbols('w2')

f = i1*w1 + i2*w2

f_sym = sympy.sympify(f)

print(f_sym)

f_an = sympy.lambdify([i1, i2, w1, w2], f_sym)

print(f_an)

w1_t = torch.tensor(3.0, requires_grad=True)
w2_t = torch.tensor(5.0, requires_grad=True)

print(f_an(10, 20, w1_t, w2_t))
print(type(f_an(10, 20, w1_t, w2_t)))


f2 = lambda j, k: f_an(j, k, w1_t, w2_t)
print(f2)

print(f2(10, 20))
print(type(f2(10, 20)))

print(w1_t.grad)
loss = f2(10, 50) - 200
loss.backward()
print(w1_t.grad)

exit()





exit()









#
