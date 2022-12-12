## ex 008.py : sarsa
import numpy as np
import gym
env = gym.make('FrozenLake-v1')
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns


def action_epsilon_greedy(q, s, epsilon=0.05):
    if np.random.rand() > epsilon:
        return np.argmax(q[s])
    return np.random.randint(4)

def evaluate_policy(q, n=500):
    acc_returns = 0
    for i in range(n):
        terminated = False
        truncated = False
        s, _ = env.reset()
        while not (terminated or truncated):
            a = action_epsilon_greedy(q, s, epsilon=0.)
            s, reward, terminated, truncated, _ = env.step(a)
            acc_returns += reward
    return acc_returns / n

def sarsa(alpha=0.02, gamma=1., epsilon=0.05, q=None, env=env):
    
    if q is None:
        q = np.zeros((16,4)).astype(np.float32)

    nb_episodes = 200000
    steps = 2000
    progress = []
    for i in tqdm(range(nb_episodes)):
        terminated = False
        truncated = False
        s, _ = env.reset()
        a = action_epsilon_greedy(q, s, epsilon=epsilon)
        while not (terminated or truncated):
            new_s, reward, terminated, truncated, _ = env.step(a)
            new_a = action_epsilon_greedy(q, new_s, epsilon=epsilon)
            q[s,a] = q[s,a] + alpha * (reward + gamma * q[new_s,new_a] - q[s,a])
            s = new_s
            a = new_a

        if i%steps == 0:
            progress.append(evaluate_policy(q, n=500))

    return q, progress

progData= pd.DataFrame()

progressList =[]    
# epList =[]
for ep in np.arange(0.01, 0.1, 0.01): ## original epsilon = 0.05
    q, progress = sarsa(alpha=0.02, epsilon=ep, gamma=0.999)
    print(evaluate_policy(q, n=10000))
    print(progress)
    #progressList.append(progress)
    progData [ep]= progress

colormap = ['blue','orange','green','red','purple','brown','pink','grey','olive','cyan']

for i in range (len(progData.columns)): 
    #print (progData.columns[i])
    plt.plot(progData.iloc[:, i], progData.index, color = colormap[i], alpha = 0.6, label = round(progData.columns[i],2))

plt.legend()
plt.show()


