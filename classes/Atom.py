import numpy as np
from math import tanh
import sympy
import os
import FileSystemTools as fst
import json
from copy import copy
import torch

'''

I guess for this one, the output_weights should probably be tuples of the form (atom, input #, weight),
because a given atom can have multiple inputs.

Likewise, the output_weights for this should probably have an entry for each output.
So it's:
-a dict for each output
-each entry of that is a dict of the index of each atom this output goes to
-each entry of that is a dict from the input index of that atom to the corresponding weight.

err, or the second two could just be combined into a (index, input) : weight dict?
Hmm, might make it harder to get all the indices easily.

inputs_received should also be a dict now, one for each atom input. Also for input_indices.



value should now be an array, one for each output.

needs:

--clear atom
--inputs, outputs
--addToInputsReceived (probably have to specify the atom AND the input index now?)


eventually: make a simple dict that has the number of times outputted to a given atom
from this atom (counting all its output indices), so you don't have to run the slow getAllChildAtoms()
every time.

eventually: more complicated atoms also don't actually need to be composed of nodes, they
should just be a huge function that's composed of all the smaller ones.

Before I was using None for the value, but we want the values to be 0 by default,
for example of atoms that get no input.

'''


class Atom:


    def __init__(self, atom_index, type, **kwargs):

        self.atom_index = atom_index
        self.type = type

        self.value = None

        self.is_input_atom = False
        self.is_output_atom = False
        self.is_bias_atom = False

        #print('creating atom (index {}) of type {}'.format(self.atom_index, self.type))
        self.loadAtomFromModuleName(self.type)


        '''if type=='Node':
            self.N_inputs = 1
            self.N_outputs = 1
        elif type=='module':
            pass
            #self.atom = EPANN()
            # Import atom properties here from json file
            #self.N_inputs = self.atom.N_inputs
            #self.N_outputs = self.atom.N_outputs'''




        # input_indices is going to be a dict, with a dict for each of its
        # inputs. Then, each of those dicts is going to be filled with entries of the form:
        # par_atom_index : [par_atom_input_index1, par_atom_input_index2, ...]
        self.input_indices = {i : {} for i in range(self.N_inputs)}

        self.inputs_received = {i : [] for i in range(self.N_inputs)}

        # output_weights is going to be a dict, with a dict for each of its
        # outputs. Then, each of those dicts is going to be filled with entries of the form:
        # child_atom_index : {child_atom_input_index1 : w1, child_atom_input_index2: w2, ...}
        self.output_weights = {i : {} for i in range(self.N_outputs)}
        self.output_weights_symbols = {}


################# setup stuff

    def setToInputAtom(self):
        self.is_input_atom = True


    def setToOutputAtom(self):
        self.is_output_atom = True


    def setToBiasAtom(self):
        self.is_bias_atom = True
        self.value = [1.0]



    def loadAtomFromModuleName(self, module_name):

        module_dir = 'atom_modules'
        ext = '.json'

        module_fname = os.path.join(module_dir, f'{module_name}{ext}')

        assert os.path.isfile(module_fname), f'File {module_fname} doesnt exist!'

        self.loadAtomFromFile(module_fname)



    def loadAtomFromFile(self, fname):

        # This loads a NN from a .json file that was saved with
        # saveNetworkToFile(). Note that it will overwrite any existing NN
        # for this object.

        with open(fname) as json_file:
            NN_dict = json.load(json_file)

        self.N_inputs = NN_dict['N_inputs']
        self.N_outputs = NN_dict['N_outputs']

        if NN_dict['Name'] == 'InputNode':
            self.setToInputAtom()

        if NN_dict['Name'] == 'OutputNode':
            self.setToOutputAtom()

        if NN_dict['Name'] == 'BiasNode':
            self.setToBiasAtom()


        self.atom_input_str = 'a_{}'
        self.input_symbols = [sympy.symbols(self.atom_input_str.format(ind)) for ind in range(self.N_inputs)]
        self.weight_symbols = []
        self.weight_strings = []

        self.atom_function_vec = NN_dict['atom_function_vec']

        # This converts it from a list of strings to a list of sympy expressions
        self.atom_function_vec = [sympy.sympify(fn) for fn in self.atom_function_vec]

        # This lambdify's them, so they can just be given arbitrary input vectors to be eval'd.
        #self.atom_fn = sympy.lambdify(self.input_symbols + , self.atom_function_vec)




################################# Getters

    def getAllInputIndices(self):
        # This just gives a list of the input indices for this node
        return(list(self.input_indices.keys()))


    def getAllOutputIndices(self):
        # This just gives a list of the output indices for this node
        return(list(self.output_weights.keys()))


    def getAllParentAtoms(self):
        # This returns only the par atom indices, not which of their output indices.
        # So it's giving the pars of all inputs, which is why it uses a set (in case
        # two inputs have the same par node)
        par_list = []
        for v in self.input_indices.values():
            if v:
                par_list += list(v.keys())

        return(list(set(par_list)))


    def getParentAtomsOfInput(self, atom_input_index):
        # Returns the par atoms of a single input index.
        # This returns only the par atom indices, not which of their output indices.
        return(list(self.input_indices[atom_input_index].keys()))


    def getAllChildAtoms(self):
        children_list = []
        for v in self.output_weights.values():
            if v:
                children_list += list(v.keys())

        return(list(set(children_list)))


    def getChildAtomsOfOutput(self, atom_input_index):
        # This returns only the child atom indices, not which of their output indices.
        return(list(self.output_weights[atom_input_index].keys()))


    def getChildAtomInputIndices(self, par_output_ind, child_atom_ind):
        # This is for, if you have the output index of this atom,
        # and the child atom you're looking at, which input indices it goes to and
        # the weights for those inputs.
        '''print('\n\nind: ', self.atom_index)
        print('ow:', self.output_weights)
        print('ow[par_out_ind]', self.output_weights[par_output_ind])'''

        return(list(self.output_weights[par_output_ind][child_atom_ind].keys()))


    def getOutputWeight(self, par_output_ind, child_atom_ind, child_atom_input_ind):

        # par_output_ind is the index of the output of this atom that we're changing.
        # child_atom_ind is the index of the atom that's being output to.
        # child_atom_input_ind is the index of the input of that atom (will be 0
        # for nodes, could be higher for more complex atoms).
        # val is if you want to set the val, std is if you want to set it to a random gaussian.
        return(self.output_weights[par_output_ind][child_atom_ind][child_atom_input_ind].item())


    def getOutputWeightSymbol(self, par_output_ind, child_atom_ind, child_atom_input_ind):
        # returns a tuple of the (str, symbol)
        return(self.output_weights_symbols[(par_output_ind, child_atom_ind, child_atom_input_ind)])


    def getOutputWeightStr(self):
        w_str = ', '.join(['{} : {}'.format(k,v) for k,v in self.output_weights.items()])
        s = '[{}]'.format(w_str)
        return(s)


################### atom modifiers

    def addToInputIndices(self, par_atom_index, par_atom_output_index, child_atom_input_index):
        #self.input_indices[input_ind].append(new_input_ind)
        # If that atom ind isn't in the output_weights dict yet, add an empty dict for it.
        if par_atom_index not in self.getParentAtomsOfInput(child_atom_input_index):
            self.input_indices[child_atom_input_index][par_atom_index] = []

        self.input_indices[child_atom_input_index][par_atom_index].append(par_atom_output_index)


    def removeFromInputIndices(self, par_atom_index, par_atom_output_index, child_atom_input_index):
        self.input_indices[child_atom_input_index][par_atom_index].remove(par_atom_output_index)
        # if that was the last output index from that par, this atom no longer gets input from
        # that par and we can remove it.
        if not self.input_indices[child_atom_input_index][par_atom_index]:
            self.input_indices[child_atom_input_index].pop(par_atom_index, None)


    def addToOutputWeights(self, par_output_index, child_atom_index, child_atom_input_index, val=None, std=0.1):

        # par_output_index is the index of the output of this atom that we're changing.
        # child_atom_index is the index of the atom that's being output to.
        # child_atom_input_index is the index of the input of that atom (will be 0
        # for nodes, could be higher for more complex atoms).
        # val is if you want to set the val, std is if you want to set it to a random gaussian.

        if val is None:
            val = np.random.normal(scale=std)

        # If that atom ind isn't in the output_weights dict yet, add an empty dict for it.
        if child_atom_index not in self.getChildAtomsOfOutput(par_output_index):
            self.output_weights[par_output_index][child_atom_index] = {}

        #self.output_weights[par_output_index][child_atom_index][child_atom_input_index] = val
        self.addOutputWeightSymbol(par_output_index, child_atom_index, child_atom_input_index)
        self.output_weights[par_output_index][child_atom_index][child_atom_input_index] = torch.tensor(val, dtype=torch.float, requires_grad=True)



    def addOutputWeightSymbol(self, par_output_index, child_atom_index, child_atom_input_index):

        w_str = 'w_{}_{}_{}_{}'.format(
        self.atom_index,
        par_output_index,
        child_atom_index,
        child_atom_input_index)

        w_symbol = sympy.symbols('w_{}_{}_{}_{}'.format(
        self.atom_index,
        par_output_index,
        child_atom_index,
        child_atom_input_index))

        self.output_weights_symbols[(par_output_index, child_atom_index, child_atom_input_index)] = (w_str, w_symbol)




    def removeFromOutputWeights(self, par_output_index, child_atom_index, child_atom_input_index):
        self.output_weights[par_output_index][child_atom_index].pop(child_atom_input_index, None)
        self.output_weights_symbols.pop((par_output_index, child_atom_index, child_atom_input_index))

        if not self.output_weights[par_output_index][child_atom_index]:
            self.output_weights[par_output_index].pop(child_atom_index, None)



    def mutateOutputWeight(self, par_output_index, child_atom_index, child_atom_input_index, std=0.1):
        with torch.no_grad():
            self.output_weights[par_output_index][child_atom_index][child_atom_input_index] += np.random.normal(scale=std)





    def set_output_weight(self, par_output_index, child_atom_index, child_atom_input_index, val):
        with torch.no_grad():
            self.output_weights[par_output_index][child_atom_index][child_atom_input_index] = torch.tensor(val, dtype=torch.float, requires_grad=True)






##################### I/O stuff

    def setInputAtomValue(self, val):
        assert self.is_input_atom, 'Can only directly set the value of an input atom! Atom {}, is_input_atom: {}'.format(self.atom_index, self.is_input_atom)
        # Uhhh I think here I'm gonna assume that only Node atoms will be inputs.
        self.value = [val]


    def getValue(self):

        # Think carefully how to do this!
        # So... I think what I'll do is, for each input ind, just sum
        # the inputs. Then it will be up to the atom itself to do a nonlinearity
        # or whatever.
        # So for a simple Node, that will happen in its forward pass.

        if self.value is not None:
            # Bias atoms should have a .value by default.
            # Input atoms should have been given a .value before this was called.
            return(self.value)
        else:

            assert not self.is_bias_atom, '.value attr must already be set with bias atom to call getValue()!'
            assert not self.is_input_atom, '.value attr must already be set with input atom to call getValue()!'

            self.value = self.forwardPass()


    def getValueOfOutputIndex(self, output_ind):

        if self.value is None:
            self.getValue()

        return(self.value[output_ind])


    def forwardPass(self):

        '''
        This will assume that the atom has already gotten all the inputs it needs
        to. You'll need to do clearAtom() on this.

        Right now this is just for figuring out the analytic form of the NN function.

        '''

        atom_input_vec = [sum(v) for v in self.inputs_received.values()]
        #print('input vec for atom {}: {}'.format(self.atom_index, atom_input_vec))
        output_vec = copy(self.atom_function_vec)
        #print('atom {} output vec before: {}'.format(self.atom_index, output_vec))
        # For each output index...
        for output_index in range(self.N_outputs):
            # Replace the atom input with the input to that input index of the atom.
            for input_index in range(self.N_inputs):
                output_vec[output_index] = output_vec[output_index].subs(self.atom_input_str.format(input_index), atom_input_vec[input_index])

        #print('atom {} output vec after: {}'.format(self.atom_index, output_vec))
        return(output_vec)


    def clearInputs(self):
        # I'm not sure why it had the if statement... is that needed for some reason??
        #if not self.is_input_atom:
        self.inputs_received = {i : [] for i in range(self.N_inputs)}


    def clearAtom(self):
        # N.B.: this clears the VALUE of the node, which is just the thing it
        # stores.

        self.clearInputs()
        if not self.is_bias_atom:
            self.value = None


    def addToInputsReceived(self, input_ind, val):
        self.inputs_received[input_ind].append(val)










''' SCRAP





    def getValue(self):

        # Think carefully how to do this!
        # So... I think what I'll do is, for each input ind, just sum
        # the inputs. Then it will be up to the atom itself to do a nonlinearity
        # or whatever.
        # So for a simple Node, that will happen in its forward pass.

        if self.value is not None:
            # Bias atoms should have a .value by default.
            # Input atoms should have been given a .value before this was called.
            return(self.value)
        else:

            assert not self.is_bias_atom, '.value attr must already be set with bias atom to call getValue()!'
            assert not self.is_input_atom, '.value attr must already be set with input atom to call getValue()!'

            self.value = self.forwardPass()

            return(self.value)

            if self.type == 'Node':
                sum_tot = [sum(v) for v in self.inputs_received.values()]
                if self.is_output_atom:
                    self.value = sum_tot
                else:
                    self.value = [self.nonlinear(s) for s in sum_tot]

                return(self.value)

            else:
                self.value = self.forwardPass()

'''















#
