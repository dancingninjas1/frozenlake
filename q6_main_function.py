from q1_frozen_lake import FrozenLake
from q2_model_based_rl import policy_iteration, value_iteration
from q3_model_free import sarsa, q_learning
from q4_non_tabular_model import LinearWrapper, linear_sarsa, linear_q_learning
from q5_deep_reinforcement_learning import FrozenLakeImageWrapper, deep_q_network_learning


def main():
    seed = 0
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
    print("Model based Algorithm")

    print('')
    print('## Policy iteration')
    policy, value = policy_iteration(env, gamma, theta, max_iterations)
    env.render(policy, value)

    print('')
    print('## Value iteration')
    policy, value = value_iteration(env, gamma, theta, max_iterations)
    env.render(policy, value)

    epsilon = 0.9
    max_episodes = 50000
    alpha = 0.85
    gamma = 0.95
    print("Model free Algorithm")
    policy, value = sarsa(env, max_episodes, alpha, gamma, epsilon)
    env.render(policy, value)

    epsilon = 0.1
    max_episodes = 50000
    alpha = 0.6
    gamma = 1.0
    policy, value = q_learning(env, max_episodes, alpha, gamma, epsilon)
    env.render(policy, value)

    print('## Linear Sarsa')
    seed = 0
    linear_env = LinearWrapper(env)
    parameters = linear_sarsa(linear_env, max_episodes, alpha, gamma, epsilon, seed=seed)
    policy, value = linear_env.decode_policy(parameters)
    linear_env.render(policy, value)

    print('## Linear Q-learning')
    parameters = linear_q_learning(linear_env, max_episodes, alpha, gamma, epsilon, seed=seed)
    policy, value = linear_env.decode_policy(parameters)
    linear_env.render(policy, value)

    print('## Deep Q-network learning')

    max_episodes = 4000
    gamma = 0.9
    image_env = FrozenLakeImageWrapper(env)
    dqn = deep_q_network_learning(image_env, max_episodes, learning_rate=0.001,
                                  gamma=gamma, epsilon=0.2, batch_size=32,
                                  target_update_frequency=4, buffer_size=256,
                                  kernel_size=3, conv_out_channels=4,
                                  fc_out_features=8, seed=4)
    policy, value = image_env.decode_policy(dqn)
    image_env.render(policy, value)


main()
