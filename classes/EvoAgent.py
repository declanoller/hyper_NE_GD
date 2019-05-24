import numpy as np
from copy import deepcopy
from classes.HyperEPANN import HyperEPANN
import FileSystemTools as fst
import matplotlib.pyplot as plt
import torch

'''

Last time, I had the EPANN class create an Agent object, but that doesn't really
make much sense -- the agent is really using the EPANN, not the other way around.
So here, EvoAgent creates a HyperEPANN object, which it uses.


So last time I had an GymAgent class and it made an env. for each one. I think that
was causing some real problems, so this time I think the smart thing to do is either
pass an env object that every EvoAgent will share, or, if nothing is passed, it will
create its own.


agent class (GymAgent or something like Walker_1D) needs members and methods:

-getStateVec()
-initEpisode()
-iterate(action) (returns (reward, state, done))
-drawState()

-state labels (list of strings corresponding to each state)
-action labels (same but with actions)
-action space type ('discrete' or 'continuous')
-N_state_terms
-N_actions



'''


class EvoAgent:

    def __init__(self, **kwargs):

        # Agent stuff
        self.agent_class = kwargs.get('agent_class', None)
        assert self.agent_class is not None, 'Need to provide an agent class! exiting'

        self.agent = self.agent_class(**kwargs)

        self.verbose = kwargs.get('verbose', False)
        self.action_space_type = self.agent.action_space_type
        self.render_type = self.agent.render_type

        self.N_inputs = self.agent.N_state_terms
        self.N_outputs = self.agent.N_actions

        # HyperEPANN stuff
        self.NN = HyperEPANN(N_inputs=self.N_inputs, N_outputs=self.N_outputs, **kwargs)

        self.NN_output_type = kwargs.get('NN_output_type', 'argmax')







############################## For interfacing with NN


    def forwardPass(self, state_vec):
        #print('state vec', state_vec)
        state_tensor = torch.tensor(state_vec, dtype=torch.float, requires_grad=False)
        output_vec = self.NN.forwardPass(state_tensor)

        if self.NN_output_type == 'argmax':
            a = self.greedyOutput(output_vec)
        else:
            a = output_vec

        return(a)


    def greedyOutput(self, vec):
        return(np.argmax(vec))


    def mutate(self, std=0.1):

        self.NN.mutate(std=std)


    def getNAtoms(self):
        return(len(self.NN.atom_list))


    def getNConnections(self):
        return(len(self.NN.weights_list))


    def plotNetwork(self, **kwargs):
        self.NN.plotNetwork(**kwargs)


    def saveNetworkAsAtom(self, **kwargs):
        self.NN.saveNetworkAsAtom(**kwargs)


    def saveNetworkToFile(self, **kwargs):
        self.NN.saveNetworkToFile(**kwargs)


    def loadNetworkFromFile(self, **kwargs):
        self.NN.loadNetworkFromFile(**kwargs)


    def clone(self):
        clone = deepcopy(self)
        return(clone)


########################### For interacting with the agent class


    def iterate(self, action):

        r, s, done, correct_answer = self.agent.iterate(action)
        #print(correct_answer)
        return(r, s, done, correct_answer)


    def initEpisode(self, **kwargs):
        grad_desc = kwargs.get('grad_desc', True)
        if grad_desc:
            if kwargs.get('reset_weights', False):
                self.NN.reinitialize_weights()
            self.NN.resetOptim(**kwargs)
        self.agent.initEpisode()


################################ For interfacing with gym env and playing


    def setMaxEpisodeSteps(self, N_steps):
        self.agent.setMaxEpisodeSteps(N_steps)



    def runEpisode(self, **kwargs):

        N_eval_steps = int(kwargs.get('N_eval_steps', 30))

        show_episode = kwargs.get('show_episode', False)
        record_episode = kwargs.get('record_episode', False)

        grad_desc = kwargs.get('grad_desc', True)

        if show_episode:
            self.createFig()
        if record_episode:
            self.agent.setMonitorOn(show_run=show_episode)

        R_tot = 0
        self.initEpisode(**kwargs)

        if grad_desc:
            N_train_steps = int(kwargs.get('N_train_steps', int(N_eval_steps*10)))
            train_iters = []
            train_curve = []

            N_batch = int(kwargs.get('N_batch', 8))
            output_batch = []
            target_batch = []

            ## Wait, do I even need to be doing clearAllAtoms() here??
            # Isn't it just doing it by equation?
            self.NN.clearAllAtoms()
            for i in range(N_train_steps):

                s = self.agent.getStateVec()
                a = self.forwardPass(s)
                self.print('s = {}, a = {}'.format(s, a))

                (r, s, done, correct_answer) = self.iterate(a.detach().numpy())

                output_batch.append(a)
                target_batch.append(torch.tensor(correct_answer, dtype=torch.float32))

                if len(output_batch)==N_batch:
                    l = self.NN.backProp(torch.stack(output_batch), torch.stack(target_batch))
                    output_batch = []
                    target_batch = []
                    train_iters.append(i)
                    train_curve.append(l)

                #l = self.NN.backProp(a, torch.tensor(correct_answer, dtype=torch.float32))

                R_tot += r

                if done:
                    #return(R_tot)
                    break

                if show_episode or record_episode:
                    self.drawState()

        # one after training, or just the usual eval step if
        # not doing gradient descent.
        R_tot = 0
        self.initEpisode()
        steps_done = 0
        self.NN.clearAllAtoms()
        for i in range(N_eval_steps):
            steps_done += 1

            s = self.agent.getStateVec()
            a = self.forwardPass(s)

            (r, s, done, correct_answer) = self.iterate(a.detach().numpy())
            R_tot += r

            if done:
                #return(R_tot)
                break

        ret = {}
        ret['r_avg'] = R_tot/steps_done
        if grad_desc:
            ret['train_iters'] = train_iters
            ret['train_curve'] = train_curve
        return(ret)



    def run_N_episodes(self, **kwargs):
        N_trials = int(kwargs.get('N_trials', 1))
        agent_mean_episode_score = 0

        for run in range(N_trials):
            ret = self.runEpisode(**kwargs)
            agent_mean_episode_score += ret['r_avg']
            if run==0 and ('train_curve' in ret.keys()):
                train_curve = ret['train_curve']

        agent_mean_episode_score = np.mean(agent_mean_episode_score)

        # Only going to return the first train curve for now.
        ret_dict = {}
        ret_dict['agent_mean_episode_score'] = agent_mean_episode_score
        if 'train_curve' in ret.keys():
            ret_dict['train_curve'] = train_curve

        return(ret_dict)


    def eval_agent_save(self, fname=None):
        all_states = self.agent.get_all_states()

        NN_output = []
        for s in all_states:
            a = self.forwardPass(s)
            NN_output.append(np.concatenate([s, a.detach().numpy()]))

        np.savetxt(fname, NN_output, delimiter='\t', fmt='%.4f')


    def drawState(self):

        if self.render_type == 'gym':
            self.agent.drawState()

        if self.render_type == 'matplotlib':
            self.agent.drawState(self.ax)
            self.fig.canvas.draw()






############################# Misc/debugging stuff


    def print(self, str):

        if self.verbose:
            print(str)



    def createFig(self):

        if self.render_type == 'matplotlib':
            self.fig, self.ax = plt.subplots(1, 1, figsize=(4,4))
            plt.show(block=False)



#
