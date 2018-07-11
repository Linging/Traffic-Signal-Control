import tensorflow as tf
import random
import numpy as np
from collections import deque
import pandas as pd
from win32com.client import Dispatch
from vis_env import VisEnv


class DQN():
    def __init__(self):
        self.replay_buffer = deque()
        self.time_step = 0
        self.epsilon = 0.9
        self.final_epsilon = 0.01

        self.action_dim = 16
        self.decay = 0.995
        self.gamma = 0.9
        self.replay_size = 2000
        self.batch_size = 32
        self.lr = 0.0001
        self.logdir = "./log"

        self.create_Q_network()
        self.create_training_method()

        self.session = tf.InteractiveSession()
        self.saver = tf.train.Saver()
        self.session.run(tf.global_variables_initializer())
        self.summary_op = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(self.logdir, self.session.graph)

    def create_Q_network(self):
        conv_W1 = self.weight_variable([3,3,2,32])
        conv_b1 = tf.random_normal([32])

        conv_W2 = self.weight_variable([3,3,32,64])
        conv_b2 = tf.random_normal([64])

        conv_W3 = self.weight_variable([3,3,64,64])
        conv_b3 = tf.random_normal([64])

        fc_W1 = self.weight_variable([120 * 64, 64])
        fc_b1 = tf.random_normal([64])

        fc_W2 = self.weight_variable([64, self.action_dim])
        fc_b2 = tf.random_normal([self.action_dim])

        self.state_input = tf.placeholder("float", [None, 8, 60, 2])

        conv_layer1 = tf.nn.relu(tf.nn.conv2d(self.state_input, conv_W1, strides=[1,1,1,1], padding='SAME') + conv_b1)
        max_pool1 = tf.nn.max_pool(conv_layer1, ksize=[1,1,2,1], strides=[1,1,2,1], padding='SAME')

        conv_layer2 = tf.nn.relu(tf.nn.conv2d(max_pool1, conv_W2, strides=[1,1,1,1], padding='SAME') + conv_b2)
        max_pool2 = tf.nn.max_pool(conv_layer2, ksize=[1,1,2,1], strides=[1,1,2,1], padding='SAME')

        conv_layer3 =  tf.nn.relu(tf.nn.conv2d(max_pool2, conv_W3, strides=[1,1,1,1], padding='SAME') + conv_b3)

        conv_layer3_flatten = tf.reshape(conv_layer3, [-1, 120 * 64])

        fc1 = tf.nn.relu(tf.matmul(conv_layer3_flatten, fc_W1) + fc_b1)
        self.Q_value = tf.matmul(fc1, fc_W2) + fc_b2

    def weight_variable(self, shape):
        initializer = tf.contrib.layers.xavier_initializer()
        var = tf.Variable(initializer(shape))
        return  var

    # ----------------------------------------------------------------------------------------------
    def create_training_method(self):
        self.action_input = tf.placeholder("float", [None, self.action_dim])
        self.y_input = tf.placeholder("float", [None])
        Q_action = tf.reduce_sum(tf.multiply(self.Q_value, self.action_input), reduction_indices = 1)
        self.cost = tf.reduce_mean(tf.square(self.y_input - Q_action))
        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.cost)
        tf.summary.scalar('cost',self.cost)

    def train_Q_network(self):
        self.time_step += 1

        minibatch = random.sample(self.replay_buffer, self.batch_size)
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        next_state_batch = [data[3] for data in minibatch]

        y_batch = []

        Q_value_batch = self.Q_value.eval(feed_dict = {self.state_input:next_state_batch})
        for i in range(0,self.batch_size):
            done = minibatch[i][4]
            if done:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(reward_batch[i] + self.gamma * np.max(Q_value_batch[i]))

        feed_dict = {
            self.y_input:y_batch,
            self.action_input:action_batch,
            self.state_input:state_batch}

        self.optimizer.run(feed_dict=feed_dict)
        summary = self.session.run(self.summary_op, feed_dict=feed_dict)
        self.writer.add_summary(summary, self.time_step)
        self.writer.flush()

    def perceive(self, state, action, reward, next_state, done):
        one_hot_action = np.zeros(self.action_dim)
        one_hot_action[action] = 1

        self.replay_buffer.append((state, one_hot_action, reward, next_state, done))
        if len(self.replay_buffer) > self.replay_size:
            self.replay_buffer.popleft()
        if len(self.replay_buffer) > self.batch_size:
            self.train_Q_network()

    def egreedy_action(self, state):
        Q_value = self.Q_value.eval(feed_dict = {self.state_input:[state]})[0]
        if random.random() <= self.epsilon:
            action_index =  random.randint(0, self.action_dim - 1)
        else:
            action_index = np.argmax(Q_value)
        if self.epsilon < self.final_epsilon:
            self.epsilon *= self.decay
        return action_index

    def action(self, state):
        Q_value = self.Q_value.eval(feed_dict = {self.state_input:[state]})[0]
        return np.argmax(Q_value)

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


import os
import random as rd

EPISODE = 20
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


dir = "./dqn/gamma=0.995"
os.makedirs(dir)
main(dir)
