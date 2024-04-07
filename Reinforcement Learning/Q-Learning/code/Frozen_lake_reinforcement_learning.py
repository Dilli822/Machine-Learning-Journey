
# Putting it Together 
# Now that we know how to do some basic things we can combine these together to create our Q- Learning algorithm,
import gym
import numpy as np
import time 
import matplotlib.pyplot as plt

env = gym.make('FrozenLake-v1')
STATES = env.observation_space.n
ACTIONS = env.action_space.n 

Q = np.zeros((STATES, ACTIONS))

EPISODES = 1500  # how many times to run the environment from the beginning
MAX_STEPS = 100  # max number of steps allowed for each run of environment
LEARNING_RATE = 0.81  # learning rate
GAMMA = 0.96  # discount factor
RENDER = False  # if you want to see training set to true 
epsilon = 0.9
rewards = []

# MAIN CORE COMPUTING HERE
for episode in range(EPISODES):
    state = env.reset()
    for _ in range(MAX_STEPS):
        if RENDER:
            env.render()
            time.sleep(0.05)  # Add a small delay to observe the environment
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state, :])
        next_state, reward, done, info = env.step(action)
        Q[state, action] = Q[state, action] + LEARNING_RATE * (reward + GAMMA * np.max(Q[next_state, :]) - Q[state, action])
        state = next_state
        if done:
            rewards.append(reward)
            epsilon -= 0.001 
            break  # reached the goal

print(Q)
avg_rewards = []
for i in range(0, len(rewards), 100):
    avg_rewards.append(sum(rewards[i:i+100]) / 100)

plt.plot(avg_rewards)
plt.ylabel('average reward')
plt.xlabel('episodes (100\'s)')
plt.title('Training Progress')
plt.show()

print(f"Average reward: {sum(rewards) / len(rewards)}")
