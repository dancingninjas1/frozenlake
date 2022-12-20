## Task 2 ##

import numpy as np

### policy iteration section ###

''' policy_evaluation function
    evaluates the given policy
    
    @param env
        the current environment
    @param policy
        the policy to be evaluated
    @param gamma
        discount factor
    @param theta
        tolerance parameter
    @param max_iterations
        the max number of iteration that can be used to retrieve the evaluation value
    
    @return value
        expected total reward starting from each game state
'''
def policy_evaluation(env, policy, gamma, theta, max_iterations):
    value = np.zeros(env.n_states, dtype=np.float)
    for _ in range(max_iterations):
        delta = 0
        for state in range(env.n_states):
            v = value[state]
            value[state] = sum([env.p(next_s, state, policy[state]) * (env.r(next_s, state, policy[state]) + gamma * value[next_s]) 
            for next_s in range(env.n_states)])
            delta = max(delta, abs(v - value[state]))
        if delta < theta:
            break
    return value

''' policy_improvement function 
    create a policy for each possible game state
    
    @param env
        the current environment
    @param value
        expected total reward starting from each game state
    @param gamma
        discount factor
    
    @return policy
        the improved policy
'''
def policy_improvement(env, value, gamma):
    policy = np.zeros(env.n_states, dtype=int)

    #TODO:
    for s in range(env.n_states):
        policy[s] = np.argmax([
            sum([env.p(next_s, s, a) * (env.r(next_s, s, a) + gamma * value[next_s])
            for next_s in range(env.n_states)])
            for a in range(env.n_actions)])
    return policy

''' policy_iteration function
    Iteratively improve the policy
    
    @param env
        the current enviroment
    @param policy
        @default = None
        the previous policy
    @param gamma
        discount factor
    @param theta
        tolerance parameter
    @param max_iterations
        the max number of iteration that can be used to retrieve the evaluation value 
    
    @return policy
        the improved policy
    @return value
        expected total reward starting from each game state
'''
def policy_iteration(env, gamma, theta, max_iterations, policy=None):
    if policy is None:
        policy = np.zeros(env.n_states, dtype=int)
    else:
        policy = np.array(policy, dtype=int)
    while(True):
        policy_initial = policy
        value = policy_evaluation(env, policy, gamma, theta, max_iterations)
        policy = policy_improvement(env, value, gamma)
        if np.array_equal(policy_initial, policy):
            break
    return policy, value


### value iteration section ###

''' value_iteration function
    iteratively increased the value and uses the value to create the policy
    
    @param env
        the current environment
    @param gamma
        discount factor
    @param theta
        tolerance parameter
    @param max_iterations
        the max number of iteration that can be used to retrieve the evaluation value
    @param value
        @default = None
        expected total reward starting from each game state
    
    @return policy
        the improved policy
    @return value
        expected total reward starting from each game state
'''
def value_iteration(env, gamma, theta, max_iterations, value=None):
    if value is None:
        value = np.zeros(env.n_states)
    else:
        value = np.array(value, dtype=np.float)

    for _ in range(max_iterations):
        delta = 0
        for state in range(env.n_states):
            v = value[state]
            value[state] = max([
            sum([env.p(s_next, state, action) * (env.r(s_next, state, action) + gamma * value[s_next]) 
            for s_next in range(env.n_states)]) 
            for action in range(env.n_actions)])
            delta = max(delta, abs(v - value[state]))
        if delta < theta:
            break

    policy = np.zeros(env.n_states, dtype=int)
    for state in range(env.n_states):
        policy[state] = np.argmax(
        [sum([env.p(s_next, state, action) * (env.r(s_next, state, action) + gamma * value[s_next])
        for s_next in range(env.n_states)])
        for action in range(env.n_actions)])

    return policy, value