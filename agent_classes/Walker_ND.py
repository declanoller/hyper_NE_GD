import matplotlib.pyplot as plt
import numpy as np
from math import ceil, floor



'''
This is the same as Walker_1D, but with N dimensions.

'''


class Walker_ND:


    def __init__(self, **kwargs):

        self.N_dims = kwargs.get('N_dims', 3)
        # This determines how discretized each dimension is.
        self.N_steps = kwargs.get('N_steps', 5)

        self.lims = np.array([-1.0, 1.0])
        self.width = np.ptp(self.lims)

        self.min_coords = np.full(self.N_dims, self.lims[0])
        self.max_coords = np.full(self.N_dims, self.lims[1])

        # This will determine the "resolution" of the grid, so a
        # smaller step size will need more steps to get to the same goal.
        self.step_size = self.width/self.N_steps

        # x, y
        self.position = np.zeros(self.N_dims)
        self.target_position = None

        self.N_state_terms = 2*self.N_dims
        self.N_actions = 2*self.N_dims
        # This will be distance of 2 units in this N-dim space.
        self.reward_thresh = np.linalg.norm(np.full(self.N_dims, 2*self.step_size))

        self.state_labels = sum([['pos_dim{}'.format(i), 'target_dim{}'.format(i)] for i in range(self.N_dims)], [])
        self.action_labels = sum([['decr_dim{}'.format(i), 'incr_dim{}'.format(i)] for i in range(self.N_dims)], [])

        empty_action_list = np.zeros(self.N_dims)
        self.action_dict = {}
        # For each dim, this assigns makes index 2*i and 2*i + 1 the [0...0] vecs,
        # but then makes the index of that dim + or - 1 (so it steps in that direction).
        for i in range(self.N_dims):

            self.action_dict[2*i] = empty_action_list
            self.action_dict[2*i + 1] = empty_action_list
            self.action_dict[2*i][i] = (-1)**(i+1)
            self.action_dict[2*i + 1][i] = (-1)**(i)


        self.action_space_type = 'discrete'
        self.render_type = 'matplotlib'


    def getStateVec(self):
        assert self.target_position is not None, 'Need target to get state vec'
        return(np.concatenate((self.position, self.target_position)))


    def initEpisode(self):
        self.resetPosition()
        self.resetTarget()


    def resetTarget(self):

        rands = np.random.rand(self.N_dims)
        self.target_position = self.lims[0] + self.width*rands


    def resetPosition(self):
        self.position = np.zeros(self.N_dims)


    def iterate(self, action):
        # maps 0,1,2,3 to L,R,D,U
        self.position += self.step_size*self.action_dict[action]
        # Keeps it in bds (maybe not necessary)
        self.position = np.minimum(np.maximum(self.position, self.min_coords), self.max_coords)
        return(self.reward(), self.getStateVec(), False)



    def reward(self):

        if np.linalg.norm(self.position - self.target_position) <= self.reward_thresh:
            self.resetTarget()
            return(1.0)
        else:
            return(-0.01)



    def drawState(self, ax):

        ax.clear()
        ax.set_xlim(tuple(self.lims))
        ax.set_ylim(tuple(self.lims))

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal')
        circle_rad = self.step_size/2

        ag = plt.Circle(self.position, circle_rad, color='tomato')
        ax.add_artist(ag)

        if self.target_position is not None:
            target = plt.Circle(self.target_position, circle_rad, color='seagreen')
            ax.add_artist(target)







#
