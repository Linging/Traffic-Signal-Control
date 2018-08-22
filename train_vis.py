from Agent.dqn_nips import DQN
from Agent.dqn_nature import DeepQNetwork
import os
import random as rd
from vis_env import VisEnv


def action_transform(a, action_dim):
    str = bin(a)[2:]
    for i in range(action_dim - len(str)):
        str = '0' + str
    str = list(str)
    for j in range(action_dim):
        str[j] = int(str[j])
    return str

EPISODE = 100
STEP = 150

def main(dir):
    # agent = DeepQNetwork(n_actions=4,n_features=2)
    agent = DQN(n_actions=16)
    env = VisEnv()
    for episode in range(0,EPISODE):
        # ==== INITIALIZE ==== #
        env.reset()
        print("Episode:",episode,"Start")

        if episode >= 30 and episode % 10 == 0: env.test = True
        state = env.state

        sum_reward = []
        env.set_flow_mode()

        for i in range(STEP):
            # ==== ACTION DECISION ==== #
            action = agent.choose_action(state)

            actions = action_transform(action, 4)

            next_state, reward, done = env.step(actions)

            sum_reward.append(reward)

            if done:
                break

            if not env.test:
                agent.store(env.state, action, reward, next_state, done)

            env.state = next_state

        ep_sum_reward = sum(sum_reward)
        print("Episode:",episode," Reward:",ep_sum_reward," steps:", i)

        if env.test:
            env.write_summary(episode, dir)


dir = "./dqn"
try:
    os.makedirs(dir)
except:
    print("Dir Exist!")

main(dir)
