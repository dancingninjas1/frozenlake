from q1_frozen_lake import FrozenLake
from q2_model_based_rl import policy_iteration, value_iteration
from q3_model_free import sarsa, q_learning
from q4_non_tabular_model import LinearWrapper, linear_sarsa, linear_q_learning
from q5_deep_reinforcement_learning import FrozenLakeImageWrapper, deep_q_network_learning
import matplotlib.pyplot as plt
import os
import numpy as np


def prepare_plot(returns, name):
    # Moving average window of length 20
    window = 20
    # Calculate the moving average using np.convolve
    mp = np.convolve(returns, np.ones(20)/20, mode='valid')
    # Plot the episode number on the x-axis and the moving average on the y-axis
    plt.clf()
    plt.cla()
    plt.plot(np.arange(1, len(mp) + 1), mp)
    plt.savefig('plots/{}.png'.format(name))
    return name


def show_all_plots(plots_):
    path = "plots/"
    # Check whether the specified path exists or not
    isExist = os.path.exists(path)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(path)
        print("Directory " + path + " created.")

    plots = [eachPlot[1] for eachPlot in plots_]
    names = [eachPlot[0] for eachPlot in plots_]

    plt.legend(plots, names)
    plt.xlabel('Episode Number')
    plt.ylabel('Moving Average')
    plt.savefig('plots/{}.png'.format("All_plots"))
    plt.show()


def main():
    seed = 0
    plots_ = []
    small_lake = [['&', '.', '.', '.'],
                  ['.', '#', '.', '#'],
                  ['.', '.', '.', '#'],
                  ['#', '.', '.', '$']]

    big_lake = [['&', '.', '.', '.', '.', '.', '.', '.'],
                ['.', '.', '.', '.', '.', '.', '.', '.'],
                ['.', '.', '.', '#', '.', '.', '.', '.'],
                ['.', '.', '.', '.', '.', '#', '.', '.'],
                ['.', '.', '.', '#', '.', '.', '.', '.'],
                ['.', '#', '#', '.', '.', '.', '#', '.'],
                ['.', '#', '.', '.', '#', '.', '#', '.'],
                ['.', '.', '.', '#', '.', '.', '.', '$']]

    lake = small_lake
    size = len(lake) * len(lake[0])
    env = FrozenLake(lake, slip=0.1, max_steps=size, seed=seed)
    gamma = 0.9
    theta = 0.001
    max_iterations = 2000
    print(" ### Model based Algorithm ### ")

    print('')
    print('## Policy iteration')
    policy, value = policy_iteration(env, gamma, theta, max_iterations)
    env.render(policy, value)

    print('')
    print('## Value iteration')
    policy, value = value_iteration(env, gamma, theta, max_iterations)
    env.render(policy, value)

    print(" ### Model free Algorithms ### ")
    epsilon = 0.9
    max_episodes = 4000
    alpha = 0.85
    gamma = 0.95
    print('## Sarsa')
    policy, value, returns_ = sarsa(env, max_episodes, alpha, gamma, epsilon)
    env.render(policy, value)
    plot_ = prepare_plot(returns_, "Sarsa")
    plots_.append(plot_)

    print('## Q-Learning')
    epsilon = 0.1
    max_episodes = 4000
    alpha = 0.6
    gamma = 1.0
    policy, value, returns_ = q_learning(env, max_episodes, alpha, gamma, epsilon)
    env.render(policy, value)
    plot_ = prepare_plot(returns_, "Q-Learning")
    plots_.append(plot_)

    print('## Linear Sarsa')
    seed = 0
    max_episodes = 4000
    linear_env = LinearWrapper(env)
    parameters, returns_ = linear_sarsa(linear_env, max_episodes, alpha, gamma, epsilon, seed=seed)
    policy, value = linear_env.decode_policy(parameters)
    linear_env.render(policy, value)
    plot_ = prepare_plot(returns_, "Linear Sarsa")
    plots_.append(plot_)

    print('## Linear Q-learning')
    max_episodes = 4000
    parameters, returns_ = linear_q_learning(linear_env, max_episodes, alpha, gamma, epsilon, seed=seed)
    policy, value = linear_env.decode_policy(parameters)
    linear_env.render(policy, value)
    plot_ = prepare_plot(returns_, "Linear-Q-Learning")
    plots_.append(plot_)

    print('## Deep Q-network learning')

    gamma = 0.9
    image_env = FrozenLakeImageWrapper(env)
    dqn = deep_q_network_learning(image_env, max_episodes, learning_rate=0.001,
                                  gamma=gamma, epsilon=0.2, batch_size=32,
                                  target_update_frequency=4, buffer_size=256,
                                  kernel_size=3, conv_out_channels=4,
                                  fc_out_features=8, seed=4)
    policy, value = image_env.decode_policy(dqn[0])
    returns_ = dqn[1]
    image_env.render(policy, value)
    plot_ = prepare_plot(returns_, "DQN")
    plots_.append(plot_)
    show_all_plots(plots_)


main()
