import gym
import minerl
from tqdm import tqdm


# Comment out to turn off logging
# import logging
# logging.basicConfig(level=logging.DEBUG)

# Making the gym environment
env = gym.make('MineRLTreechop-v0')
obs = env.reset()
done = False
print("Reset Successful!")

loop = tqdm(range(20000))
for t in loop:
    loop.set_description(f"Testing Prolonged Stepping")
    action = env.action_space.sample()
    obs, reward, done, _ = env.step(action)
    # print(f"reward: {reward}")
    # print(f"action: {action['camera']}")
    # print("-----")
    env.render() # Comment out to turn off rendering


