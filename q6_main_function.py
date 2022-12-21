from q1_frozen_lake import FrozenLake
from q2_model_based_rl import policy_iteration
from q3_model_free import sarsa, q_learning


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

    env = FrozenLake(small_lake, slip=0.1, max_steps=16, seed=seed)
    gamma = 0.9
    theta = 0.001
    max_iterations = 2000
    print("Model based Algorithm")
    policy, value = policy_iteration(env, gamma, theta, max_iterations)
    env.render(policy, value)

    epsilon = 0.9
    max_episodes = 10000
    alpha = 0.85
    gamma = 0.95
    print("Model free Algorithm")
    policy, value = sarsa(env, max_episodes, alpha, gamma, epsilon)
    env.render(policy, value)

    epsilon = 0.1
    max_episodes = 10000
    alpha = 0.6
    gamma = 1.0
    policy, value = q_learning(env, max_episodes, alpha, gamma, epsilon)
    env.render(policy, value)
main()
