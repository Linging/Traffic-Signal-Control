from Agent.pg import PolicyGradient
import numpy as np
import tensorflow as tf
from vis_env import VisEnv
import os

def summarize(reward, i, summary_writer, tag):
  summary = tf.Summary()
  summary.value.add(tag=tag, simple_value=reward)
  summary_writer.add_summary(summary, i)
  summary_writer.flush()

def action_transform(a, action_dim):
    str = bin(a)[2:]
    for i in range(action_dim - len(str)):
        str = '0' + str
    str = list(str)
    for j in range(action_dim):
        str[j] = int(str[j])
    return str

EPISODE = 1000
STEPS = 150

def train(dir):
    env = VisEnv()

    RL = PolicyGradient(
        n_actions=16,
        learning_rate=0.001,
        reward_decay=0.9,
    )
    for i_episode in range(EPISODE):

        env.reset()

        env.set_flow_mode()

        if i_episode % 5 == 0: env.test = True

        for i in range(STEPS):

            observation = env.state
            action = RL.choose_action(observation)
            actions = action_transform(action, RL.n_actions)

            observation_, reward, done = env.step(actions)

            RL.store_transition(observation, action, reward)
            if done:
                break
            env.state = observation_


        ep_rs_sum = sum(RL.ep_rs)
        print("episode:", i_episode, "  reward:", ep_rs_sum," steps:", i)
        summarize(ep_rs_sum,i_episode,RL.writer,'train')
        vt = RL.learn(i_episode)

        if env.test:
            env.write_summary(i_episode, dir)


dir = "./pg/reward_func_1"
os.makedirs(dir)
train(dir)
