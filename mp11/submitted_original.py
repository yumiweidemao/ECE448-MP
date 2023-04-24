""" This file contains the EC deep_q learner without state quantization (to be used with test_model_7600.pkl). """
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn


class q_learner():
    def __init__(self, alpha, epsilon, gamma, nfirst, state_cardinality):
        '''
        Create a new q_learner object.
        Your q_learner object should store the provided values of alpha,
        epsilon, gamma, and nfirst.
        It should also create a Q table and an N table.
        Q[...state..., ...action...] = expected utility of state/action pair.
        N[...state..., ...action...] = # times state/action has been explored.
        Both are initialized to all zeros.
        Up to you: how will you encode the state and action in order to
        define these two lookup tables?  The state will be a list of 5 integers,
        such that 0 <= state[i] < state_cardinality[i] for 0 <= i < 5.
        The action will be either -1, 0, or 1.
        It is up to you to decide how to convert an input state and action
        into indices that you can use to access your stored Q and N tables.
        
        @params:
        alpha (scalar) - learning rate of the Q-learner
        epsilon (scalar) - probability of taking a random action
        gamma (scalar) - discount factor        
        nfirst (scalar) - exploring each state/action pair nfirst times before exploiting
        state_cardinality (list) - cardinality of each of the quantized state variables

        @return:
        None
        '''
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.nfirst = nfirst
        self.state_cardinality = state_cardinality

        totalNumOfStates = state_cardinality[0] * state_cardinality[1] * state_cardinality[2] * \
                           state_cardinality[3] * state_cardinality[4]
        self.Q = np.zeros(shape=(totalNumOfStates * 3))
        self.N = np.zeros(shape=(totalNumOfStates * 3))

    def idx(self, state, action):
        # return the index to self.Q and self.N given a state and an action.
        idx_state = state[0] * self.state_cardinality[1] * self.state_cardinality[2] * \
                    self.state_cardinality[3] * self.state_cardinality[4] + \
                    state[1] * self.state_cardinality[2] * self.state_cardinality[3] * self.state_cardinality[4] + \
                    state[2] * self.state_cardinality[3] * self.state_cardinality[4] + \
                    state[3] * self.state_cardinality[4] + state[4]
        idx_action = action + 1
        return idx_state * 3 + idx_action

    def report_exploration_counts(self, state):
        '''
        Check to see how many times each action has been explored in this state.
        @params:
        state (list of 5 ints): ball_x, ball_y, ball_vx, ball_vy, paddle_y.
          These are the (x,y) position of the ball, the (vx,vy) velocity of the ball,
          and the y-position of the paddle, all quantized.
          0 <= state[i] < state_cardinality[i], for all i in [0,4].

        @return:
        explored_count (array of 3 ints): 
          number of times that each action has been explored from this state.
          The mapping from actions to integers is up to you, but there must be three of them.
        '''
        counts = [0, 0, 0]
        actions = [-1, 0, 1]
        for i in range(3):
            action = actions[i]
            counts[i] = int(self.N[self.idx(state, action)])
        return counts

    def choose_unexplored_action(self, state):
        '''
        Choose an action that has been explored less than nfirst times.
        If many actions are underexplored, you should choose uniformly
        from among those actions; don't just choose the first one all
        the time.
        
        @params:
        state (list of 5 ints): ball_x, ball_y, ball_vx, ball_vy, paddle_y.
           These are the (x,y) position of the ball, the (vx,vy) velocity of the ball,
          and the y-position of the paddle, all quantized.
          0 <= state[i] < state_cardinality[i], for all i in [0,4].

        @return:
        action (scalar): either -1, or 0, or 1, or None
          If all actions have been explored at least n_explore times, return None.
          Otherwise, choose one uniformly at random from those w/count less than n_explore.
          When you choose an action, you should increment its count in your counter table.
        '''
        action_counts = self.report_exploration_counts(state)
        valid_actions = []
        for i in range(3):
            if action_counts[i] < self.nfirst:
                valid_actions.append(i - 1)
        if not valid_actions:
            return None
        choice = np.random.choice(valid_actions)
        self.N[self.idx(state, choice)] += 1
        return choice

    def report_q(self, state):
        '''
        Report the current Q values for the given state.
        @params:
        state (list of 5 ints): ball_x, ball_y, ball_vx, ball_vy, paddle_y.
          These are the (x,y) position of the ball, the (vx,vy) velocity of the ball,
          and the y-position of the paddle, all quantized.
          0 <= state[i] < state_cardinality[i], for all i in [0,4].

        @return:
        Q (array of 3 floats): 
          reward plus expected future utility of each of the three actions. 
          The mapping from actions to integers is up to you, but there must be three of them.
        '''
        Q = []
        for action in [-1, 0, 1]:
            Q.append(self.Q[self.idx(state, action)])
        return Q

    def q_local(self, reward, newstate):
        '''
        The update to Q estimated from a single step of game play:
        reward plus gamma times the max of Q[newstate, ...].
        
        @param:
        reward (scalar float): the reward achieved from the current step of game play.
        newstate (list of 5 ints): ball_x, ball_y, ball_vx, ball_vy, paddle_y.
          These are the (x,y) position of the ball, the (vx,vy) velocity of the ball,
          and the y-position of the paddle, all quantized.
          0 <= state[i] < state_cardinality[i], for all i in [0,4].
        
        @return:
        Q_local (scalar float): the local value of Q
        '''
        Q_local = reward + self.gamma * max(self.Q[self.idx(newstate, action=-1)],
                                            self.Q[self.idx(newstate, action=0)],
                                            self.Q[self.idx(newstate, action=1)])
        return Q_local

    def learn(self, state, action, reward, newstate):
        '''
        Update the internal Q-table on the basis of an observed
        state, action, reward, newstate sequence.
        
        @params:
        state: a list of 5 numbers: ball_x, ball_y, ball_vx, ball_vy, paddle_y.
          These are the (x,y) position of the ball, the (vx,vy) velocity of the ball,
          and the y-position of the paddle.
        action: an integer, one of -1, 0, or +1
        reward: a reward; positive for hitting the ball, negative for losing a game
        newstate: a list of 5 numbers, in the same format as state
        
        @return:
        None
        '''
        Q_local = self.q_local(reward, newstate)
        self.Q[self.idx(state, action)] += self.alpha * (Q_local - self.Q[self.idx(state, action)])

    def save(self, filename):
        '''
        Save your Q and N tables to a file.
        This can save in any format you like, as long as your "load" 
        function uses the same file format.  We recommend numpy.savez,
        but you can use something else if you prefer.
        
        @params:
        filename (str) - filename to which it should be saved
        @return:
        None
        '''
        np.savez(filename, Q=self.Q, N=self.N)

    def load(self, filename):
        '''
        Load the Q and N tables from a file.
        This should load from whatever file format your save function
        used.  We recommend numpy.load, but you can use something
        else if you prefer.
        
        @params:
        filename (str) - filename from which it should be loaded
        @return:
        None
        '''
        npzfile = np.load(filename)
        self.Q = npzfile['Q']
        self.N = npzfile['N']

    def exploit(self, state):
        '''
        Return the action that has the highest Q-value for the current state, and its Q-value.
        @params:
        state (list of 5 ints): ball_x, ball_y, ball_vx, ball_vy, paddle_y.
          These are the (x,y) position of the ball, the (vx,vy) velocity of the ball,
          and the y-position of the paddle, all quantized.
          0 <= state[i] < state_cardinality[i], for all i in [0,4].

        @return:
        action (scalar int): either -1, or 0, or 1.
          The action that has the highest Q-value.  Ties can be broken any way you want.
        Q (scalar float): 
          The Q-value of the selected action
        '''
        Q_values = [
            self.Q[self.idx(state, action=-1)],
            self.Q[self.idx(state, action=0)],
            self.Q[self.idx(state, action=1)]
        ]
        action = np.argmax(Q_values) - 1
        Q = np.max(Q_values)
        return action, Q

    def act(self, state):
        '''
        Decide what action to take in the current state.
        If any action has been taken less than nfirst times, then choose one of those
        actions, uniformly at random.
        Otherwise, with probability epsilon, choose an action uniformly at random.
        Otherwise, choose the action with the best Q(state,action).
        
        @params: 
        state: a list of 5 integers: ball_x, ball_y, ball_vx, ball_vy, paddle_y.
          These are the (x,y) position of the ball, the (vx,vy) velocity of the ball,
          and the y-position of the paddle, all quantized.
          0 <= state[i] < state_cardinality[i], for all i in [0,4].
       
        @return:
        -1 if the paddle should move upward
        0 if the paddle should be stationary
        1 if the paddle should move downward
        '''
        unexplored_action = self.choose_unexplored_action(state)
        if unexplored_action is not None:
            return unexplored_action

        random_number = np.random.uniform(0, 1)
        if random_number < self.epsilon:
            return np.random.choice([-1, 0, 1])

        best_action, _ = self.exploit(state)
        return best_action

class ShittyModel(nn.Module):
    def __init__(self):
        '''
        A model containing three difference multiperceptrons for the three actions.
        Different nets are used to account for the relationship between actions & states.
        Have tried a single net with input size 6 (5 state + 1 action), doesn't work well.
        '''
        super(ShittyModel, self).__init__()

        layer1_size = 64
        layer2_size = 32
        layer3_size = 16
        # neural net for action=-1
        self.net1 = nn.Sequential(
            nn.Linear(5, layer1_size),
            nn.ReLU(),
            nn.Linear(layer1_size, layer2_size),
            nn.ReLU(),
            nn.Linear(layer2_size, layer3_size),
            nn.ReLU(),
            nn.Linear(layer3_size, 1)
        )
        # neural net for action=0
        self.net2 = nn.Sequential(
            nn.Linear(5, layer1_size),
            nn.ReLU(),
            nn.Linear(layer1_size, layer2_size),
            nn.ReLU(),
            nn.Linear(layer2_size, layer3_size),
            nn.ReLU(),
            nn.Linear(layer3_size, 1)
        )
        # neural net for action=1
        self.net3 = nn.Sequential(
            nn.Linear(5, layer1_size),
            nn.ReLU(),
            nn.Linear(layer1_size, layer2_size),
            nn.ReLU(),
            nn.Linear(layer2_size, layer3_size),
            nn.ReLU(),
            nn.Linear(layer3_size, 1)
        )

    def forward(self, state, action):
        if action == -1:
            return self.net1(torch.Tensor(state))
        if action == 0:
            return self.net2(torch.Tensor(state))
        if action == 1:
            return self.net3(torch.Tensor(state))
        return None


class deep_q():
    def __init__(self, alpha, epsilon, gamma, nfirst):
        '''
        Create a new deep_q learner.
        Your q_learner object should store the provided values of alpha,
        epsilon, gamma, and nfirst.
        It should also create a deep learning model that will accept
        (state,action) as input, and estimate Q as the output.
        
        @params:
        alpha (scalar) - learning rate of the Q-learner
        epsilon (scalar) - probability of taking a random action
        gamma (scalar) - discount factor
        nfirst (scalar) - exploring each state/action pair nfirst times before exploiting

        @return:
        None
        '''
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.nfirst = nfirst

        self.model = ShittyModel()
        self.loss_fn = torch.nn.MSELoss()

        # define three optimizers for net1, net2, net3
        learning_rate = 1e-6
        self.optimizer1 = torch.optim.SGD(params=self.model.net1.parameters(), lr=learning_rate)
        self.optimizer2 = torch.optim.SGD(params=self.model.net2.parameters(), lr=learning_rate)
        self.optimizer3 = torch.optim.SGD(params=self.model.net3.parameters(), lr=learning_rate)

    def act(self, state):
        '''
        Decide what action to take in the current state.
        You are free to determine your own exploration/exploitation policy -- 
        you don't need to use the epsilon and nfirst provided to you.
        
        @params: 
        state: a list of 5 floats: ball_x, ball_y, ball_vx, ball_vy, paddle_y.
       
        @return:
        -1 if the paddle should move upward
        0 if the paddle should be stationary
        1 if the paddle should move downward
        '''
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice([-1, 0, 1])

        with torch.no_grad():
            q1 = self.model.forward(state, action=-1).item()
            q2 = self.model.forward(state, action=0).item()
            q3 = self.model.forward(state, action=1).item()

        action = np.argmax([q1, q2, q3]) - 1
        return action

    def learn(self, state, action, reward, newstate):
        with torch.no_grad():
            q1 = self.model.forward(newstate, -1).item()
            q2 = self.model.forward(newstate, 0).item()
            q3 = self.model.forward(newstate, 1).item()
            q_local = torch.Tensor([reward + self.gamma * max(q1, q2, q3)])

        q = self.model.forward(state, action)
        loss = self.loss_fn(q, q_local)
        if action == -1:
            self.optimizer1.zero_grad()
            loss.backward()
            self.optimizer1.step()
        elif action == 0:
            self.optimizer2.zero_grad()
            loss.backward()
            self.optimizer2.step()
        elif action == 1:
            self.optimizer3.zero_grad()
            loss.backward()
            self.optimizer3.step()

    def save(self, filename):
        '''
        Save your trained deep-Q model to a file.
        This can save in any format you like, as long as your "load" 
        function uses the same file format.
        
        @params:
        filename (str) - filename to which it should be saved
        @return:
        None
        '''
        torch.save(self.model.state_dict(), filename)

    def load(self, filename, train=False):
        '''
        Load your deep-Q model from a file.
        This should load from whatever file format your save function
        used.
        
        @params:
        filename (str) - filename from which it should be loaded
        @return:
        None
        '''
        self.model.load_state_dict(torch.load(filename))
        if not train:
            self.model.eval()

    def report_q(self, state):
        with torch.no_grad():
            q = [self.model.forward(state, action=-1).item(),
                 self.model.forward(state, action=0).item(),
                 self.model.forward(state, action=1).item()]
        return q
