from Agent.dqn_nips import DQN
from Agent.dqn_nature import DeepQNetwork
import os
import random as rd
from vis_env import VisEnv
import tensorflow as tf


def action_transform(a, action_dim):
    str = bin(a)[2:]
    for i in range(action_dim - len(str)):
        str = '0' + str
    str = list(str)
    for j in range(action_dim):
        str[j] = int(str[j])
    return str

def summarize(reward, i, summary_writer, tag):
  summary = tf.Summary()
  summary.value.add(tag=tag, simple_value=reward)
  summary_writer.add_summary(summary, i)
  summary_writer.flush()


EPISODE = 10000
STEP = 300

def main(Agent, csv_summary):

    agent = Agent
    env = VisEnv()
    for episode in range(0,EPISODE):
        # ==== INITIALIZE ==== #
        env.reset()
        env.random_seed(7)
        print("Episode:",episode,"Start")

        if episode >= 900 and episode % 10 == 0: env.test = True
        state = env.init_state

        sum_reward = []
        env.set_flow_mode()

        for i in range(STEP):
            # ==== ACTION DECISION ==== #
            # action = agent.choose_action(state, pre_actions)
            action = agent.choose_action(state)
            actions = action_transform(action, 4)

            next_state, reward, done = env.step(actions)

            sum_reward.append(reward)

            if done:
                break

            info = actions

            if not env.test:
                agent.store(state, action, reward, next_state, done, info)

            state = next_state


        ep_sum_reward = sum(sum_reward)
        print("Episode:",episode," Reward:",ep_sum_reward," steps:", i, " Epsilon:", agent.epsilon)

        summarize(ep_sum_reward, episode, agent.writer, 'Rewards')

        if env.test:
            env.write_summary(episode, csv_summary)

for learn_rate in [0.00025]:
    for replay_size in [10000]:
        for batch_size in [32]:
            dir = "lr=" + str(learn_rate) + " rep=" \
                  + str(replay_size) + " bat=" + str(batch_size)
            csv_summary = "./dqn/" + dir
            tensorboard_logs = "./logs/" + dir
            try:
                os.makedirs(csv_summary)
            except:
                print("Dir Exist!")

            # Agent = DQN(16,learning_rate=learn_rate,memory_size=replay_size,
            #             batch_size=batch_size, tensorboard_logs=tensorboard_logs)
            Agent = DeepQNetwork(n_actions=16)

            main(Agent, csv_summary)
