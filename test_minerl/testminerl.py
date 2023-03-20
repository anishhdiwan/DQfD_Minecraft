import gym
import minerl

# Comment out to turn off logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Making the gym environment
env = gym.make('MineRLTreechop-v0')
obs = env.reset()
done = False
print("Reset Successful!")

# Printing some shapes
# print(f"observation: {obs}")
# print(obs.shape)

for i in range(5):
    action = env.action_space.sample()
    obs, reward, done, _ = env.step(action)
    print(f"reward: {reward}")
    print(f"action: {action['camera']}")
    print("-----")
    env.render() # Comment out to turn off rendering