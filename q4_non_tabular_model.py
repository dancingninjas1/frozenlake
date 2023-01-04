import numpy as np
from matplotlib import pyplot as plt


class LinearWrapper:
    def __init__(self, env):
        self.env = env

        self.n_actions = self.env.n_actions
        self.n_states = self.env.n_states
        self.n_features = self.n_actions * self.n_states

    def encode_state(self, s):
        features = np.zeros((self.n_actions, self.n_features))
        for a in range(self.n_actions):
            i = np.ravel_multi_index((s, a), (self.n_states, self.n_actions))
            features[a, i] = 1.0

        return features

    def decode_policy(self, theta):
        policy = np.zeros(self.env.n_states, dtype=int)
        value = np.zeros(self.env.n_states)
        for s in range(self.n_states):
            features = self.encode_state(s)
            q = features.dot(theta)

            policy[s] = np.argmax(q)
            value[s] = np.max(q)

        return policy, value

    def reset(self):
        return self.encode_state(self.env.reset())

    def step(self, action):
        state, reward, done = self.env.step(action)

        return self.encode_state(state), reward, done

    def render(self, policy=None, value=None):
        self.env.render(policy, value)


def choose_action(env, epsilon, i, q, random_state):
    if i < env.n_actions:
        a = i
    else:
        a = get_action(random_state, epsilon, i, env, q)
    return a


def linear_sarsa(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)
    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)
    returns_ = []
    theta = np.zeros(env.n_features)
    for episode in range(max_episodes):
        features = env.reset()
        q = features.dot(theta)
        a = choose_action(env, epsilon, episode, q, random_state)
        done = False
        while not done:
            state_s, r, done = env.step(a)
            state_a = get_action(random_state, epsilon, episode, env, q)

            delta = r - q[a]
            q = state_s.dot(theta)
            delta += (gamma * q[state_a])
            theta += eta[episode] * delta * features[a]
            features = state_s

            a = state_a
        returns_ += [env.decode_policy(theta)[1].mean()]
    return theta, returns_


def linear_q_learning(env, max_episodes, eta, gamma, epsilon, seed=None):
    print("")
    random_state = np.random.RandomState(seed)
    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)

    theta = np.zeros(env.n_features)
    j = 0
    returns_ = []
    for episode in range(max_episodes):
        features = env.reset()
        q = features.dot(theta)
        done = False
        while not done:
            a = choose_action(env, epsilon, episode, q, random_state)
            j += 1

            state_s, reward_pre, done = env.step(a)

            delta = reward_pre - q[a]
            q = state_s.dot(theta)
            delta += (gamma * max(q))
            theta += eta[episode] * delta * features[a]
            features = state_s
        returns_ += [env.decode_policy(theta)[1].mean()]
    return theta, returns_


def get_action(random_state, epsilon, i, env, q):
    if random_state.random(1) < epsilon[i]:
        state_a = random_state.choice(range(env.n_actions))
    else:
        state_a = random_state.choice(np.array(np.argwhere(q == np.amax(q))).flatten(), 1)[0]
    return state_a
