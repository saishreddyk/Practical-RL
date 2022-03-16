import gym
import numpy as np

# Load Environment and structure of Q-table
env = gym.make('FrozenLake8x8-v1')
Q = np.zeros([env.observation_space.n, env.action_space.n]) # .n gives no of states and actions
print(Q.shape)

print("Observation space")
print(env.observation_space.sample())
print("",end="\n\n\n")
print("Action space")
print(env.action_space.sample())
print("",end="\n\n\n")
print("Q-Table")
print(Q)

# Defining hyperparameters of Q-Learning
lr = 0.628  # learning rate (alpha)
gamma = 0.9 # discount factor
epochs = 5000 # also episodes
rev_list = [] # reward storage

# for i in range(epochs):
#     print("Running episode {}".format(i))
#     s = env.reset() # first, we gotta reset the environment
#     rAll = 0
#     d = False
#     j = 0 # we do a max of 100 actions in each episode, prolonging this will have negative effects
#     while j < 100:
#         print("Running {} iteration of {} episode".format(j, i))
#         env.render()
#         j += 1
#         # choose a max action from Q table
#         a = np.argmax(Q[s,:] + np.random.randn(1, env.action_space.n)*(1./(i+1)))
#         # get new state and reward from environment
#         s1, r, d, _  = env.step(a) # returns observation, reward, if_done?, dict(for info)
#         # now update q table
#         Q[s, a] = Q[s, a] + lr*(r + gamma*np.argmax(Q[s1,:]) - Q[s, a])
#         rAll += r
#         s = s1
#         if d == True:
#             break
#     rev_list.append(rAll)
#     env.render()

# print("Reward sum on all episodes "+ str(sum(rev_list)/epochs))
# print("Final Values of Q-Table")
# print(Q)