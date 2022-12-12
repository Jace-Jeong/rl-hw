
import numpy as np
import gym
env = gym.make("FrozenLake-v0")

def compute_value_function(policy , gamma ):

    num_iterations = 1000
    threshold = 1e-20
    #gamma = 1.0
    
    value_table = np.zeros(env.observation_space.n)

    for i in range(num_iterations):
        updated_value_table = np.copy(value_table)

        for s in range(env.observation_space.n):
            a = policy[s]

            value_table[s] = sum(
                [prob * (r + gamma * updated_value_table[s_])
                    for prob, s_, r, _ in env.P[s][a]])

        if np.sum(np.fabs(updated_value_table - value_table)) <= threshold:
            break

    return value_table

def extract_policy(value_table, gamma):

    #gamma = 0.99
    policy = np.zeros(env.observation_space.n)
    for s in range(env.observation_space.n):

        Q_values = [sum([prob*(r + gamma * value_table[s_]) 
                             for prob, s_, r, _ in env.P[s][a]]) 
                                   for a in range(env.action_space.n)]

        policy[s] = np.argmax(np.array(Q_values))
    
    return policy

def policy_iteration(env, gamma):

    num_iterations = 1000
    
    policy = np.zeros(env.observation_space.n)

    for i in range(num_iterations):
        
        value_function = compute_value_function(policy, gamma )

        new_policy = extract_policy(value_function, gamma)

        if np.all(policy == new_policy):
            stopCnt = i
            break

        policy = new_policy 

    return policy, stopCnt



def myrange(start, end, step):
    r = start
    while(r<end):
        yield r
        r += step
myrangeList = []
for i in myrange(0,1,0.1):
    myrangeList.append(i)

iterList = []
for gamma in myrange(0,1,0.1) : 
    optimal_policy , iterCnt = policy_iteration(env, gamma)
    iterList.append(iterCnt)
    
    print(optimal_policy)

plt.scatter(myrangeList, iterList, c = 'green')
plt.xlabel("gamma")
plt.ylabel("timesteps")
plt.show()
