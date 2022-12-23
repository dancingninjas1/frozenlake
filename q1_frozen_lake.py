from abc import ABC
from itertools import product

import numpy as np
import contextlib


# Print the value matrix
@contextlib.contextmanager
def _printoptions(*args, **kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    try:
        yield
    finally:
        np.set_printoptions(**original)


class EnvironmentModel:
    """Abstract class of a model of an environment

    Attributes:
        n_states: number of states
        n_actions: number of actions
        seed: seed that controls the pseudorandom number generator
    """

    def __init__(self, n_states, n_actions, seed=None):
        self.n_states = n_states
        self.n_actions = n_actions
        self.random_state = np.random.RandomState(seed)

    """Get probability of transitioning from state to next state given action
        Need to implement in child class

        Args:
            next_state:
                next state to transition
            state:
                current state
            action:
                action to take at current state
    
        Returns:
            Probability of transitioning from state to next state given action
    """

    def p(self, next_state, state, action):
        raise NotImplementedError()

    """Get expected reward in having transitioned from state to next state given action
        Need to implement in child class

        Args:
            next_state:
                next state to transition
            state:
                current state
            action:
                action to take at current state

        Returns:
            Expected reward in having transitioned from state to next state given action
    """

    def r(self, next_state, state, action):
        raise NotImplementedError()

    """Draw the next state

    Args:
        state:
            current state
        action:
            action to take at current state

    Returns:
        A state drawn according to p together with the corresponding expected reward
    """

    def draw(self, state, action):
        p = [self.p(ns, state, action) for ns in range(self.n_states)]
        next_state = self.random_state.choice(self.n_states, p=p)
        reward = self.r(next_state, state, action)

        return next_state, reward


class Environment(EnvironmentModel):
    """Subclass of EnvironmentModel, represents an interactive environment

    Attributes:
        n_states: number of states
        n_actions: number of actions
        max_steps: maximum number of steps for interaction
        pi: probability distribution over initial states
        seed: seed that controls the pseudorandom number generator
    """

    def __init__(self, n_states, n_actions, max_steps, pi, seed=None):
        EnvironmentModel.__init__(self, n_states, n_actions, seed)

        self.max_steps = max_steps

        self.pi = pi
        if self.pi is None:
            self.pi = np.full(n_states, 1. / n_states)

    """Restarts the interaction between the agent and the environment
        
    Returns:
        State which drawn according to the probability distribution over initial states
    """

    def reset(self):
        self.n_steps = 0
        self.state = self.random_state.choice(self.n_states, p=self.pi)

        return self.state

    """Step forward and draw the next state

    Args:
        action:
            action to take at current state

    Returns:
        A next state drawn according to p, the corresponding expected reward, and a flag variable
    
    Raises:
      Exception: action is invalid
    """

    def step(self, action):
        if action < 0 or action >= self.n_actions:
            raise Exception('Invalid action.')

        self.n_steps += 1
        done = (self.n_steps >= self.max_steps)

        self.state, reward = self.draw(self.state, action)

        return self.state, reward, done

    """Render function
        Need to implement in child class
    Args:
        policy:
            policy matrix, default: None
        value:
            value matrix, default: None
    """

    def render(self, policy=None, value=None):
        raise NotImplementedError()


class FrozenLake(Environment):
    """Child class of Environment, represents the frozen lake environment

    Attributes:
        n_states: number of states
        n_actions: number of actions
        max_steps: maximum number of steps for interaction
        pi: probability distribution over initial states
        seed: seed that controls the pseudorandom number generator
        lake: a matrix that represents a lake
        slip: probability that the agent will slip at any given time step
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

        # absorbing_state
        self.absorb_state_idx = n_states - 1
        self.absorb_state = tuple((-1, -1))

        Environment.__init__(self, n_states, n_actions, max_steps, pi, seed)

        # Set of actions: up, left, down, right
        self.actions = [(-1, 0), (0, -1), (1, 0), (0, 1)]

        # self.p_file = np.load('p.npy')

        # state index of a lake
        iter_state_product = list(product(range(self.lake.shape[0]), range(self.lake.shape[1])))
        iter_state_product.append(self.absorb_state)  # add absorb state
        self.state_dict = {s: i for (i, s) in enumerate(iter_state_product)}

        # initiate the transition probability
        self.transition_probability = np.zeros((self.n_states, self.n_states, self.n_actions))

        for state, state_idx in self.state_dict.items():
            for action_idx, action in enumerate(self.actions):

                # current state is absorb state
                if state_idx == self.absorb_state_idx:
                    next_state = self.absorb_state
                    next_state_idx = self.state_dict.get(next_state)
                    self.transition_probability[next_state_idx, state_idx, action_idx] = 1

                # current state is goal or starting point
                elif self.lake_flat[state_idx] == '$' or self.lake_flat[state_idx] == '#':
                    next_state = self.absorb_state
                    next_state_idx = self.state_dict.get(next_state)
                    self.transition_probability[next_state_idx, state_idx, action_idx] = 1

                # current state is normal state
                else:
                    # set transition_probability for current state
                    next_state = (state[0] + action[0], state[1] + action[1])
                    next_state_idx = self.state_dict.get(next_state, state_idx)
                    # base transition probability: 1.0 - self.slip
                    self.transition_probability[next_state_idx, state_idx, action_idx] += 1.0 - self.slip

                    # set transition_probability for slip action
                    for a in self.actions:
                        n_s = (state[0] + a[0], state[1] + a[1])
                        n_s_i = self.state_dict.get(n_s, state_idx)
                        # additional slip transition probabilities self.slip / 4
                        self.transition_probability[n_s_i, state_idx, action_idx] += self.slip / 4

    """Frozen lake step function that calls the step function of the parent

        Args:
            action:
                action to take at current state

        Returns:
            A next state drawn according to p, the corresponding expected reward, and a flag variable
    """

    def step(self, action):
        state, reward, done = Environment.step(self, action)
        done = (state == self.absorb_state_idx) or done
        return state, reward, done

    """Get probability of transitioning from state to next state given action

        Args:
            next_state:
                next state to transition
            state:
                current state
            action:
                action to take at current state

        Returns:
            Expected reward in having transitioned from state to next state given action
    """

    def p(self, next_state, state, action):
        # return self.p_file[next_state, state, action]
        return self.transition_probability[next_state, state, action]

    """Get expected reward in having transitioned from state to next state given action

        Args:
            next_state:
                next state to transition
            state:
                current state
            action:
                action to take at current state

        Returns:
            Expected reward in having transitioned from state to next state given action
    """

    def r(self, next_state, state, action):
        char = 'xs'
        if (state < self.n_states - 1): char = self.lake_flat[state]  # if not in the absorbing state
        if (char == '$'): return 1  # but on goal then return reward one
        return 0  # for any other action no reward

    """Render the decision matrix

        Args:
            policy:
                policy matrix, default: None
            value:
                value matrix, default: None
    """

    def render(self, policy=None, value=None):
        if policy is None:
            lake = np.array(self.lake_flat)

            if self.state < self.absorb_state_idx:
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

    """Play function
    """

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
