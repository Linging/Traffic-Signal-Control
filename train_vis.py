from dqn_nips import DQN
from dqn_nature import DeepQNetwork
import os
import random as rd
from vis_env import VisEnv


def action_transform(a):
    switch = {
        0: [0, 0, 0, 0], 1: [0, 0, 0, 1],
        2: [0, 0, 1, 0], 3: [0, 1, 0, 0],
        4: [1, 0, 0, 0], 5: [0, 0, 1, 1],
        6: [0, 1, 0, 1], 7: [1, 0, 0, 1],
        8: [0, 1, 1, 0], 9: [1, 0, 1, 0],
        10: [1, 1, 0, 0], 11: [0, 1, 1, 1],
        12: [1, 1, 1, 0], 13: [1, 0, 1, 1],
        14: [1, 1, 0, 1], 15: [1, 1, 1, 1]
    }
    return switch[a]

EPISODE = 200
STEP = 150

def main(dir):
    agent = DQN()

    env = VisEnv()
    for episode in range(0,EPISODE):
        # ==== INITIALIZE ==== #
        env.reset()
        print("Episode:",episode,"Start")

        if episode >= 14 and episode % 2 == 0: env.test = True
        state = env.state

        sum_reward = []

        env.set_flow_mode()
        for i in range(STEP):
            # ==== ACTION DECISION ==== #
            action = agent.egreedy_action(state)

            actions = action_transform(action)

            next_state, reward, down = env.step(actions)

            sum_reward.append(reward)

            if down:
                break

            if not env.test:
                agent.perceive(env.state, action, reward, next_state, down)

            env.state = next_state

        ep_sum_reward = sum(sum_reward)
        print("Episode:",episode," Reward:",ep_sum_reward," steps:", i)

        if env.test:
            env.write_summary(episode, dir)


dir = "./dqn"
os.makedirs(dir)
main(dir)
