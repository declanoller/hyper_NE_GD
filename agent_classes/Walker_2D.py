import matplotlib.pyplot as plt
import numpy as np

'''
This is the same as Walker_1D, but with 2 dimensions.

'''


class Walker_2D:


    def __init__(self, **kwargs):

        self.lims = np.array([-1.0, 1.0])
        self.width = self.lims[1] - self.lims[0]

        # This will determine the "resolution" of the grid, so a
        # smaller step size will need more steps to get to the same goal.
        self.step_size = self.width/10.0

        # x, y
        self.position = np.array([0,0])
        self.target_position = None

        self.N_state_terms = 4
        self.N_actions = 4

        self.state_labels = ['pos_x', 'pos_y', 'pos_target_x', 'pos_target_y']
        self.action_labels = ['L', 'R', 'D', 'U']

        self.action_dict = {
        0 : np.array([-1, 0]),
        1 : np.array([1, 0]),
        2 : np.array([0, -1]),
        3 : np.array([0, 1]),
        }

        self.action_space_type = 'discrete'
        self.render_type = 'matplotlib'


    def getStateVec(self):
        assert self.target_position is not None, 'Need target to get state vec'
        return(np.concatenate((self.position, self.target_position)))


    def initEpisode(self):
        self.resetPosition()
        self.resetTarget()


    def resetTarget(self):

        x = np.random.random()
        y = np.random.random()
        self.target_position = np.array([self.lims[0] + self.width*x, self.lims[0] + self.width*y])
        # print('new target pos: {:.3f}'.format(self.target_position))


    def resetPosition(self):
        self.position = np.zeros(2)


    def iterate(self, action):
        # maps 0,1,2,3 to L,R,D,U
        self.position += self.step_size*self.action_dict[action]
        # Keeps it in bds (maybe not necessary)
        self.position[0] = min(max(self.position[0], self.lims[0]), self.lims[1])
        self.position[1] = min(max(self.position[1], self.lims[0]), self.lims[1])
        
        '''self.position[0] = min(max(self.position[0], self.lims[0] + self.step_size), self.lims[1] - self.step_size)
        self.position[1] = min(max(self.position[1], self.lims[0] + self.step_size), self.lims[1] - self.step_size)'''
        return(self.reward(), self.getStateVec(), False)



    def reward(self):

        if np.linalg.norm(self.position - self.target_position) <= 2.1*self.step_size:
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
