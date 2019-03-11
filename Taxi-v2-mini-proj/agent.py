import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.alpha = 0.1
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay_rate = 0.999
        self.min_epsilon = 0.00001

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        self.epsilon = np.max([self.min_epsilon, self.epsilon*self.epsilon_decay_rate])
        do_exploration = np.random.uniform(0,1) < self.epsilon
        if do_exploration:
            action = np.random.randint(0, self.nA)
        else:
            action = np.argmax(self.Q[state])
           
        return action

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        self.Q[state][action] += self.alpha * (reward + self.gamma * (np.max(self.Q[next_state]) - self.Q[state][action]))