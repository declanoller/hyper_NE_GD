import matplotlib.pyplot as plt
import numpy as np


class LogicAgentXor:


    def __init__(self, **kwargs):


        self.truth_table_dict = {
        (0,0) : np.array([0]),
        (0,1) : np.array([1]),
        (1,0) : np.array([1]),
        (1,1) : np.array([0]),
        }

        self.N_state_terms = 2
        self.N_actions = 1
        self.current_state_ind = 0
        self.N_state_vals = 2**self.N_state_terms
        self.getNewState()

        self.state_labels = ['x', 'y']
        self.action_labels = ['TF']


        self.action_space_type = 'discrete'
        self.render_type = 'matplotlib'
        self.getStateVec()





    def getStateVec(self):
        #self.state_vec = np.array(list(self.truth_table_dict.keys())[self.current_state_ind % self.N_state_vals])
        return(self.state_vec)


    def getNewState(self):
        # Get a 2-tuple of 0's, 1's
        # This will make it just cycle through the truth_table_dict keys, instead of
        # trying random non deterministic shit.

        self.state_vec = np.random.random_integers(0, 1, self.N_state_terms)
        self.getStateVec()


    def initEpisode(self):
        self.getNewState()
        self.getStateVec()



    def iterate(self, action):
        # Action 0 is go L, action 1 is go R.
        r, correct_answer = self.reward(action)
        self.getNewState()
        #return(r, self.getStateVec(), self.current_state_ind == self.N_state_vals, correct_answer)
        return(r, self.getStateVec(), False, correct_answer)



    def reward(self, action):
        # Here, action is the truth value for the state_vec of
        # truth values to the truth table.
        # So we take the difference, and then take the negative of that
        # because we want it to minimize that diff. in general.
        correct_answer = self.truth_table_dict[tuple(self.state_vec)]
        return(-np.linalg.norm(correct_answer - action), correct_answer)



    def drawState(self, ax):

        ax.clear()
        pass







#
