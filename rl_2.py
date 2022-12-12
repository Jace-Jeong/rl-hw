# ex002.py
import numpy as np
import gym
env = gym.make('FrozenLake-v0', is_slippery=False)

def value_iteration(env, gamma ):

    num_iterations = 2000
    threshold = 1e-20
    #gamma = 0.99

    value_table = np.zeros(env.observation_space.n)

    for i in range(num_iterations):
        
        updated_value_table = np.copy(value_table)

        for s in range(env.observation_space.n):

            Q_values = [sum([prob*(r + gamma * updated_value_table[s_]) 
                             for prob, s_, r, _ in env.P[s][a]]) 
                                   for a in range(env.action_space.n)]

            value_table[s] = max(Q_values)

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


avgList = []
gamList = []

def evaluate_policy(policy, gamma ):
    num_episodes = 1000
    num_timesteps = 1000
    total_reward = 0
    total_timestep = 0

    for i in range(num_episodes):
        state = env.reset()
        for t in range(num_timesteps):
            if policy is None:
                action = env.action_space.sample()
            else:
                action = policy[state]
                # state -> t로 수정이 맞을듯?
                #print(state[0])
                #print(type(state[0]))
                #action = policy[state[0]]
            #observation, reward, terminated, truncated, info ( 기존 코드에 truncated 추가)


            state, reward, done  ,info  = env.step(action)
            #print("state", state)
            total_reward += reward

            if done:
                break

        total_timestep += t

    gamList.append(gamma)    
    avgList.append(total_timestep/num_episodes)
    #print("Number of successful episodes: %d / %d"%(total_reward, num_episodes))
    #print("Average number of timesteps per episode: %.2f"%(total_timestep/num_episodes))

def myrange(start, end, step):
    r = start
    while(r<end):
        yield r
        r += step

myrangeList = []


for gam in myrange(0.0,1,0.11) : 
    optimal_value_function = value_iteration(env, gam )
    optimal_policy = extract_policy(optimal_value_function, gam)
    print("gamma : ", gam)
    print(optimal_policy)
    #evaluate_policy(None,gam)
    evaluate_policy(optimal_policy, gam)


plt.scatter(gamList, avgList, c = 'green')
plt.xlabel("gamma")
plt.ylabel("average number of timesteps per episode")
plt.show()
