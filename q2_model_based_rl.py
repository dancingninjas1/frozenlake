import numpy as np

"""policy evaluation function
    
    Args:
        env:
            the environment to learn
        policy:
            the policy Pi to evaluate
        gamma:
            discount factor
        theta:
            tolerance
        max_iterations:
            the max number of iteration 

    Returns:
        expected total reward
"""


def policy_evaluation(env, policy, gamma, theta, max_iterations):
    # initialize arbitrary V(s) array
    v_s = np.zeros(env.n_states, dtype=np.float)
    i = 0
    delta = theta
    while (i < max_iterations) and (delta >= theta):
        delta = 0
        i += 1
        for s in range(env.n_states):
            # assign value to v
            v = v_s[s]
            # calculate V(s)
            v_s[s] = sum(
                [env.p(next_s, s, policy[s]) * (env.r(next_s, s, policy[s]) + gamma * v_s[next_s])
                 for next_s in range(env.n_states)])
            # assign value to delta
            delta = max(delta, abs(v - v_s[s]))
    return v_s


"""policy improvement function 
    to get the max Qpi
    
    Args:
        env:
            the environment to learn
        value:
            policy evaluation result
        gamma:
            discount factor

    Returns:
        expected total reward starting from each game state
"""


def policy_improvement(env, value, gamma):
    # initiate policy array to be arbitrary (all zero)
    policy = np.zeros(env.n_states, dtype=int)

    for s_idx in range(env.n_states):
        # find Pi prime for each state
        policy[s_idx] = np.argmax([
            sum([env.p(n_s_idx, s_idx, a) * (env.r(n_s_idx, s_idx, a) + gamma * value[n_s_idx])
                 for n_s_idx in range(env.n_states)])
            for a in range(env.n_actions)])
    return policy


"""policy iteration function
    iteratively increased the value and uses the value to create the policy
    
    Args:
        env:
            the environment to learn
        gamma:
            discount factor
        theta:
            tolerance parameter
        max_iterations:
            the max number of iteration that can be used to retrieve the evaluation value
        policy:
            the previous policy
        
    Returns:
        improved policy and expected total reward starting from each game state
"""


def policy_iteration(env, gamma, theta, max_iterations, policy=None):
    # initialize arbitrary policy
    if policy is None:
        policy = np.zeros(env.n_states, dtype=int)
    else:
        policy = np.array(policy, dtype=int)

    while True:
        previous_policy = policy
        # one step policy improvement
        value = policy_evaluation(env, policy, gamma, theta, max_iterations)
        policy = policy_improvement(env, value, gamma)
        # if convergence, then break the iteration
        if np.array_equal(previous_policy, policy):
            break
    return policy, value


"""value_iteration function
    iteratively increased the value and uses the value to create the policy
    
    Args:
        env:
            the environment to learn
        gamma
            discount factor
        theta
            tolerance parameter
        max_iterations
            the max number of iteration that can be used to retrieve the evaluation value
        value
            expected total reward starting from each game state
    
    Returns:
        improved policy and expected total reward starting from each game state
"""


def value_iteration(env, gamma, theta, max_iterations, value=None):
    # initialize arbitrary V(s) array
    if value is None:
        value = np.zeros(env.n_states)
    else:
        value = np.array(value, dtype=np.float)

    # create sequence V0, V1, ... V*
    i = 0
    delta = theta
    while (i < max_iterations) and (delta >= theta):
        delta = 0
        i += 1
        for s in range(env.n_states):
            v = value[s]
            value[s] = max([
                sum([env.p(next_s, s, a) * (env.r(next_s, s, a) + gamma * value[next_s])
                     for next_s in range(env.n_states)])
                for a in range(env.n_actions)])
            delta = max(delta, abs(v - value[s]))

    # find optimal policy with V*(s)
    policy = np.zeros(env.n_states, dtype=int)
    for s in range(env.n_states):
        policy[s] = np.argmax(
            [sum([env.p(next_s, s, a) * (env.r(next_s, s, a) + gamma * value[next_s])
                  for next_s in range(env.n_states)])
             for a in range(env.n_actions)])

    return policy, value
