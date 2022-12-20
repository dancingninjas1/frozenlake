from abc import ABC
from itertools import product

import numpy as np
import contextlib

"""Gets and prints the spreadsheet's header columns

Args:
    table_handle:
        An open smalltable.Table instance.

Returns:
    A dict mapping keys to the corresponding table row data fetched.
"""
@contextlib.contextmanager
def _printoptions(*args, **kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    try:
        yield
    finally:
        np.set_printoptions(**original)


class EnvironmentModel:
    """Summary of class here.

    Attributes:
        n_states:
        n_actions:
        random_state:
    """

    def __init__(self, n_states, n_actions, seed=None):
        self.n_states = n_states
        self.n_actions = n_actions

        self.random_state = np.random.RandomState(seed)

    """Gets and prints the spreadsheet's header columns

        Args:
            table_handle:
                An open smalltable.Table instance.

        Returns:
            A dict mapping keys to the corresponding table row data fetched.
    """
    def p(self, next_state, state, action):
        raise NotImplementedError()

    """Gets and prints the spreadsheet's header columns

        Args:
            table_handle:
                An open smalltable.Table instance.

        Returns:
            A dict mapping keys to the corresponding table row data fetched.
    """
    def r(self, next_state, state, action):
        raise NotImplementedError()

    """One step dynamic

    Args:
        table_handle:
            An open smalltable.Table instance.

    Returns:
        A dict mapping keys to the corresponding table row data fetched.
    """
    def draw(self, state, action):
        p = [self.p(ns, state, action) for ns in range(self.n_states)]
        next_state = self.random_state.choice(self.n_states, p=p)
        reward = self.r(next_state, state, action)

        return next_state, reward


class Environment(EnvironmentModel):
    """Summary of class here.

    Attributes:
        n_states:
        n_actions:
        random_state:
    """

    def __init__(self, n_states, n_actions, max_steps, pi, seed=None):
        EnvironmentModel.__init__(self, n_states, n_actions, seed)

        self.max_steps = max_steps

        self.pi = pi
        if self.pi is None:
            self.pi = np.full(n_states, 1. / n_states)

    """Gets and prints the spreadsheet's header columns

    Args:
        table_handle:
            An open smalltable.Table instance.

    Returns:
        A dict mapping keys to the corresponding table row data fetched.
    """
    def reset(self):
        self.n_steps = 0
        self.state = self.random_state.choice(self.n_states, p=self.pi)

        return self.state

    """Gets and prints the spreadsheet's header columns

    Args:
        table_handle:
            An open smalltable.Table instance.

    Returns:
        A dict mapping keys to the corresponding table row data fetched.
    """
    def step(self, action):
        if action < 0 or action >= self.n_actions:
            raise Exception('Invalid action.')

        self.n_steps += 1
        done = (self.n_steps >= self.max_steps)

        self.state, reward = self.draw(self.state, action)

        return self.state, reward, done

    """Gets and prints the spreadsheet's header columns

    Args:
        table_handle:
            An open smalltable.Table instance.

    Returns:
        A dict mapping keys to the corresponding table row data fetched.
    """

    def render(self, policy=None, value=None):
        raise NotImplementedError()


class FrozenLake(Environment):
    """Summary of class here.

    Attributes:
        n_states:
        n_actions:
        random_state:
    """

    def __init__(self, lake, slip, max_steps, seed=None):
        """
        lake: A matrix that represents the lake. For example:
         lake =  [['&', '.', '.', '.'],
                  ['.', '#', '.', '#'],
                  ['.', '.', '.', '#'],
                  ['#', '.', '.', '$']]
        slip: The probability that the agent will slip
        max_steps: The maximum number of time steps in an episode
        seed: A seed to control the random number generator (optional)
        """

        # start (&), frozen (.), hole (#), goal ($)
        self.lake = np.array(lake)
        self.lake_flat = self.lake.reshape(-1)

        self.slip = slip

        n_states = self.lake.size + 1
        n_actions = 4

        pi = np.zeros(n_states, dtype=float)
        pi[np.where(self.lake_flat == '&')[0]] = 1.0

        self.absorbing_state = n_states - 1

        Environment.__init__(self, n_states, n_actions, max_steps, pi, seed)

        self.actions = [(-1, 0), (0, -1), (1, 0), (0, 1)]

        self.dict_states = {p: i for (i, p) in enumerate(product(range(self.lake.shape[0]), range(self.lake.shape[1])))}

        self.transition_probability_file = np.load('p.npy')
        n_s, s, _ = self.transition_probability_file.shape

        """
        self.transition_popability = np.zeros((self.n_states, self.n_actions, self.n_states))  # applicable to all lakes

        for current_state, current_state_index in self.dict_states.items():
            for n, action in enumerate(self.actions):
                next_state = (current_state[0] + action[0], current_state[1] + action[1])
                index_next_state = self.dict_states.get(next_state, 'NaN')
                current_tile = self.lake_flat[current_state_index]

                if index_next_state == 'NaN':
                    index_next_state = current_state_index

                if current_tile == '$' or current_tile == '#':  # hole or goal
                    self.transition_popability[
                        current_state_index, n, self.absorbing_state] = 1  # 0.925 #1.0 - self.slip + (self.slip/self.n_actions)
                    continue
                else:
                    self.transition_popability[
                        current_state_index, n, index_next_state] = 0.925  # 1.0 - self.slip + (self.slip/self.n_actions)

                for n_slip, action_slip in enumerate(self.actions):
                    if n_slip == n:
                        continue
                    else:
                        slip_state = (current_state[0] + action_slip[0], current_state[1] + action_slip[1])
                        index_slip_state = self.dict_states.get(slip_state, 'NaN')
                        if index_slip_state == 'NaN':
                            self.transition_popability[
                                current_state_index, n, current_state_index] += 0.025  # self.slip/self.n_actions
                        if not index_slip_state == 'NaN':
                            self.transition_popability[
                                current_state_index, n, index_slip_state] = 0.025  # self.slip/self.n_actions
        """

    ''' step function
        calls the step function of the parent

        @param action
            the next action

        @return self.state
            return the current state
        @return reward
            return the current reward
        @return done
            bool if the game max step limit has been reached 
    '''

    def step(self, action):
        state, reward, done = Environment.step(self, action)
        done = (state == self.absorbing_state) or done
        return state, reward, done

    ''' p function
        probability to be returned for combination of next state, state, and action

        @param next_state
            the next game state
        @param state
            current game state
        @param action
            action taken

        @return p
            probability
    '''
    def p(self, next_state, state, action):
        return self.transition_probability_file[next_state, state, action]
        # return self.transition_popability[state, action, next_state]

    ''' r function
        expected reward in having transitioned from state to next state given action

        @param next_state
            the next game state
        @param state
            current game state
        @param action
            action taken

        @return r
            expected reward based on parameters
    '''

    def r(self, next_state, state, action):
        char = 'xs'
        if (state < self.n_states - 1): char = self.lake_flat[state]  # if not in the absorbing state
        if (char == '$'): return 1  # but on goal then return reward one
        return 0  # for any other action no reward

    ''' render function
        renders and prints the state

        @param policy
            @default = None 
            the current policy used
        @param value
            @default = None
            expected total reward starting from each game state
    '''

    def render(self, policy=None, value=None):
        if policy is None:
            lake = np.array(self.lake_flat)

            if self.state < self.absorbing_state:
                lake[self.state] = '@'

            print(lake.reshape(self.lake.shape))
        else:
            # UTF-8 arrows look nicer, but cannot be used in LaTeX
            # https://www.w3schools.com/charsets/ref_utf_arrows.asp
            actions = ['↑', '←', '↓', '→']

            print('Lake:')
            print(self.lake)

            print('Policy:')
            policy = np.array([actions[a] for a in policy[:-1]])
            print(policy.reshape(self.lake.shape))

            print('Value:')
            with _printoptions(precision=3, suppress=True):
                print(value[:-1].reshape(self.lake.shape))

    ''' play function
        lets the user play the game
    '''

    def play(self):
        actions = ['w', 'a', 's', 'd']

        state = self.reset()
        self.render()

        done = False
        while not done:
            c = input('\nMove: ')
            if c not in actions:
                raise Exception('Invalid action')

            state, r, done = self.step(actions.index(c))

            self.render()
            print('Reward: {0}.'.format(r))
