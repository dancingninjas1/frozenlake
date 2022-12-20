from q1_frozen_lake import FrozenLake
from q2_model_based_rl import policy_iteration


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
    print(' # Model−based algorithms ')
    print(' ')
    policy, value = policy_iteration(env, gamma, theta, max_iterations)
    env.render(policy, value)


main()
