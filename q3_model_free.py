import numpy as np
from matplotlib import pyplot as plt
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def get_action(random_state, epsilon, index, env, q, s):
    if random_state.random(1) < epsilon[index]:
        state_a = random_state.choice(range(env.n_actions))
    else:
        state_a = random_state.choice(np.array(np.argwhere(q[s] == np.amax(q[s]))).flatten(), 1)[0]
    return state_a


def choose_action(env, episode, epsilon, q, random_state, s):
    if episode < env.n_actions:
        a = episode
    else:
        a = get_action(random_state, epsilon, episode, env, q, s)
    return a


def sarsa(env, max_episodes, eta, gamma, epsilon, seed=None):
    print("Sarsa Learning")
    random_state = np.random.RandomState(seed)
    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)

    q = np.zeros((env.n_states, env.n_actions))

    iterations = []
    for episode in range(max_episodes):
        s = env.reset()
        done = False
        a = choose_action(env, episode, epsilon, q, random_state, s)
        while not done:
            state_s, r, done = env.step(a)

            state_a = get_action(random_state, epsilon, episode, env, q, s)

            q[s, a] += eta[episode] * ((r + gamma * q[state_s, state_a]) - q[s, a])
            a, s = state_a, state_s
        iterations += [q.max(axis=1).mean()]
    mp = np.convolve(iterations, np.ones(20) / 20, mode='valid')
    #plt.clf()
    #plt.cla()
    #plt.plot(np.arange(1, len(mp) + 1), mp)
    #plt.savefig('plots/sarsa.png')
    policy = q.argmax(axis=1)
    value = q.max(axis=1)

    return policy, value


def q_learning(env, max_episodes, eta, gamma, epsilon, seed=None):
    print("Q Learning")
    random_state = np.random.RandomState(seed)
    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)

    q = np.zeros((env.n_states, env.n_actions))

    for episode in range(max_episodes):
        s = env.reset()
        j = 0
        done = False
        while not done:
            a = choose_action(env, episode, epsilon, q, random_state, s)
            j += 1

            state_s, reward_pre, done = env.step(a)

            q[s, a] += eta[episode] * (reward_pre + gamma * max(q[state_s]) - q[s, a])
            s = state_s

    policy = q.argmax(axis=1)
    value = q.max(axis=1)

    return policy, value
