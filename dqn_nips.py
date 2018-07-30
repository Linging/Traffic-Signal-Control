import tensorflow as tf
import random
import numpy as np
from replay_memory import *

class DQN():
    def __init__(self,
                n_actions,
                learning_rate=1e-4,
                reward_decay=0.9,
                e_greedy=0.9,
                replace_target_iter=300,
                memory_size=500,
                batch_size=32,
                e_greedy_increment=None,
                output_graph=False,
                dueling=False,
                ):
        self.time_step = 0
        self.epsilon = e_greedy
        self.final_epsilon = 0.01

        self.action_dim = n_actions
        self.decay = 0.995
        self.gamma = reward_decay
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.lr = learning_rate
        self.logdir = "./dqn_log"

        self.dueling = False

        self.create_Q_network()
        self.create_training_method()

        self.session = tf.InteractiveSession()
        self.saver = tf.train.Saver()
        self.session.run(tf.global_variables_initializer())
        self.summary_op = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(self.logdir, self.session.graph)

        self.experience_replay = ReplayMemoryFast(memory_size, batch_size)

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

        if self.dueling:
            # Dueling DQN
            with tf.variable_scope('Value'):
                w1 = tf.get_variable('dueling_w2', [64, 1], initializer=w_initializer)
                b1 = tf.get_variable('dueling_b2', [1, 1], initializer=b_initializer)
                self.V = tf.matmul(fc1, w1) + b1

            with tf.variable_scope('Advantage'):
                w2 = tf.get_variable('dueling_w2', [64, self.action_dim], initializer=w_initializer)
                b2 = tf.get_variable('dueling_b2', [1, self.action_dim], initializer=b_initializer)
                self.A = tf.matmul(fc1, w2) + b2

            with tf.variable_scope('Q'):
                Q_value = self.V + (self.A - tf.reduce_mean(self.A, axis=1, keep_dims=True))     # Q = V(s) + A(s,a)
        else:
            with tf.variable_scope('Q'):
                w2 = tf.get_variable('w2', [64, self.action_dim], initializer=w_initializer)
                b2 = tf.get_variable('b2', [1, self.action_dim], initializer=b_initializer)
                Q_value = tf.matmul(fc1, w2) + b2

        return Q_value

    def weight_variable(self, shape):
        initializer = tf.contrib.layers.xavier_initializer()
        var = tf.Variable(initializer(shape))
        return  var

    # ----------------------------------------------------------------------------------------------
    def create_training_method(self):
        self.action_input = tf.placeholder("float", [None, self.action_dim])
        self.y_input = tf.placeholder("float", [None])
        Q_action = tf.reduce_sum(tf.multiply(self.Q_value, self.action_input), reduction_indices = 1)

        Q_mean = tf.reduce_mean(self.Q_value, axis=1)
        tf.summary.scalar('Q_mean', Q_mean)

        self.cost = tf.reduce_mean(tf.square(self.y_input - Q_action))
        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.cost)
        tf.summary.scalar('cost',self.cost)

    def learn(self):
        self.time_step += 1

        minibatch = self.experience_replay.sample()

        if len(minibatch) == 0:
            return

        batch_s = np.asarray([data[0] for data in minibatch])
        actions = np.asarray([data[1] for data in minibatch])
        batch_r = np.asarray([data[2] for data in minibatch])
        batch_s_ = np.asarray([data[3] for data in minibatch])

        batch_a = np.zeros([self.batch_size, self.action_dim])
        for i in range(self.batch_size):
            batch_a[i, actions[i]] = 0
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
            self.action_input:batch_a,
            self.state_input:batch_s}

        self.optimizer.run(feed_dict=feed_dict)
        summary = self.session.run(self.summary_op, feed_dict=feed_dict)
        self.writer.add_summary(summary, self.time_step)
        self.writer.flush()

	def store(self, state, action, reward, next_state, is_terminal):
		# rewards clipping
		if self.reward_clipping > 0.0:
			reward = np.clip(reward, -self.reward_clipping, self.reward_clipping)

		self.experience_replay.store(state, action, reward, next_state, is_terminal)

    def choose_action(self, state):
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
