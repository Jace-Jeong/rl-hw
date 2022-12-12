import numpy as np
import gym
from tqdm import tqdm
import matplotlib as plt
import matplotlib.pyplot as plt 

# Q learning params
#ALPHA = 0.1 # learning rate 1211 주석
#GAMMA = 0.99 # reward discount 1211 주석
LEARNING_COUNT = 100000
TEST_COUNT = 1000
#EPS = 0.1 ## epsilon 

TURN_LIMIT = 100
#ALPHA = 0.1 # learning rate
#GAMMA = 0.99 # reward discount


class Agent:
    def __init__(self, env , alpha, gamma , eps):
        self.env = env
        self.episode_reward = 0.0
        self.q_val = np.zeros(16 * 4).reshape(16, 4).astype(np.float32)
        self.ALPHA = alpha
        self.GAMMA = gamma 
        self.EPS = eps
        print("ALPHA" , alpha)
        print("GAMMA" , gamma)

    def learn(self):
        # one episode learning
        state, _ = self.env.reset()
        
        for t in range(TURN_LIMIT):
            if np.random.rand() < self.EPS: # explore
                act = self.env.action_space.sample() # random
            else: # exploit
                act = np.argmax(self.q_val[state])
            next_state, reward, terminated, truncated, info = self.env.step(act)
            q_next_max = np.max(self.q_val[next_state])
            # Q <- Q + a(Q' - Q)
            # <=> Q <- (1-a)Q + a(Q')
            self.q_val[state][act] = (1 - self.ALPHA) * self.q_val[state][act]\
                                 + self.ALPHA * (reward + self.GAMMA * q_next_max)
            
            if terminated or truncated:
                return reward
            else:
                state = next_state
        return 0.0 # over limit

    def test(self):
        state, _ = self.env.reset()
        for t in range(TURN_LIMIT):
            act = np.argmax(self.q_val[state])
            next_state, reward, terminated, truncated, info = self.env.step(act)
            if terminated or truncated:
                return reward
            else:
                state = next_state
        return 0.0 # over limit



def main():
    env = gym.make("FrozenLake-v1")
    
    
    for eps in np.arange(0.1, 0.9, 0.1): 
        alpha = 0.1 
        gamma = 0.99 
        #for gamma in np.arange(1, 0.1, -0.1):
            #agent = Agent(env)
            #alpha = alpha
            #gamma = gamma 
        agent = Agent(env,alpha, gamma, eps)
            #alpList = []
            #ist = []
            
                
    lAlpList = []
    lGamList = []
    lTotRewardList=[]
    lAvgRewardList =[]
    lQValList=[]
    lEpsList = []

    tAlpList = []
    tGamList = []
    tTotRewardList=[]
    tAvgRewardList =[]
    tQValList=[]
    tEpsList = []

    
    #for alpha in np.arange(0, 1, 0.1): 
    for eps in np.arange(0.1, 0.9, 0.1):
            alpha = 0.1
            gamma = 0.99
    
            print("###### LEARNING #####")
            reward_total = 0.0
            for i in tqdm(range(LEARNING_COUNT)):
                reward_total += agent.learn()
            print("episodes      : {}".format(LEARNING_COUNT))
            print("total reward  : {}".format(reward_total))
            print("average reward: {:.2f}".format(reward_total / LEARNING_COUNT))
            print("Q Value       :{}".format(agent.q_val))

            lAlpList.append(alpha)
            lGamList.append(gamma)
            lTotRewardList.append(reward_total)
            lAvgRewardList.append(reward_total / LEARNING_COUNT)
            lQValList.append(agent.q_val)
            lEpsList.append(eps)
            

            print("###### TEST #####")
            reward_total = 0.0
            for i in tqdm(range(TEST_COUNT)):
                reward_total += agent.test()
            print("episodes      : {}".format(TEST_COUNT))
            print("total reward  : {}".format(reward_total))
            print("average reward: {:.2f}".format(reward_total / TEST_COUNT))
            tAlpList.append(alpha)
            tGamList.append(gamma)
            tTotRewardList.append(reward_total)
            tAvgRewardList.append(reward_total / LEARNING_COUNT)
            tQValList.append(agent.q_val)
            tEpsList.append(eps)

   



    plt.scatter(lEpsList, lAvgRewardList, c = 'green')
    plt.xlabel("epsilon")
    plt.ylabel("avg Reward")
    plt.show()

    plt.scatter(tEpsList, tAvgRewardList, c = 'blue')
    plt.xlabel("epsilon")
    plt.ylabel("avg Reward")
    plt.show()   
 
            

if __name__ == "__main__":
    main()
