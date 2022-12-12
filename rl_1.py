import matplotlib as plt
import matplotlib.pyplot as plt 
from tqdm import tqdm ## tqdm : 진행사항 보여주는 라이브러리 ! 
import gym

env = gym.make("FrozenLake-v0")

arr_avg_timesteps = []
arr_timesteps = []
for i in range (1,50) : 
    arr_timesteps.append(i)

    num_episodes = 100
    #num_timesteps = 50
    num_timesteps = i
    total_reward = 0
    total_timestep = 0

    for i in tqdm(range(num_episodes)):
        state = env.reset()
        
        for t in range(num_timesteps):
            random_action = env.action_space.sample()
            #observation, reward, terminated, truncated, info

            new_state, reward, done , info = env.step(random_action)
            total_reward += reward

            if done:
                break

        total_timestep += t
    #print("Number of successful episodes: %d / %d"%(total_reward, num_episodes))
    #print("Average number of timesteps per episode: %.2f"%(total_timestep/num_episodes))
    arr_avg_timesteps.append(total_timestep/num_episodes)

plt.scatter(arr_avg_timesteps, arr_timesteps, c = 'green')
plt.xlabel("avg timesteps")
plt.ylabel("timesteps")
plt.show()
