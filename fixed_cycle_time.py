from vis_env import VisEnv
import os

STEP = 150

env = VisEnv()
env.reset()

a1 = [0,0,0,1]
a2 = [1,1,1,0]

env.set_flow_mode()

for i in range(STEP):

    env.test = True

    if i % 6 < 3:
        actions = a1
    else:
        actions = a2

    next_state, reward, down = env.step(actions)

dir = "./fixed_cycle"
os.makedirs(dir)
if env.test:
    env.write_summary(1, dir)
