import numpy as np
import tensorflow as tf
from Agent.replay_memory import *
from Agent.tf_utils import *
from Agent.networks import *

np.random.seed(1)
tf.set_random_seed(1)


# Deep Q Network off-policy
class DeepQNetwork:
    def __init__(
            self,
            n_actions,
            learning_rate=0.01,
            reward_decay=0.8,
            e_greedy_max=0.99,
            replace_target_iter=300,
            memory_size=10000,
            batch_size=16,
            e_greedy_increment=0.0001,
            output_graph=True,
            dueling=False,
            state_size = [84,84],
    ):
        self.n_actions = n_actions
        self.state_size = state_size
        self.lr = learning_rate
        self.gamma = reward_decay
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_max = e_greedy_max
        self.replace_target_iter = replace_target_iter
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.dueling = dueling
        # total learning step
        self.learn_step_counter = 0
        self.network = QNetworkNature(state_size, n_actions, 'eval_net')
        self.target_network = QNetworkNature(state_size, n_actions, 'target_net')
        self.reward_clipping = 10

        # initialize replay memory
        self.experience_replay = ReplayMemoryFast(memory_size, batch_size)

        # consist of [target_net, evaluate_net]
        self._build_net()

        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')

        with tf.variable_scope('soft_replacement'):
            self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
        self.sess = tf.Session(config=config)

        self.summary_op = tf.summary.merge_all()

        if output_graph:
            # $ tensorboard --logdir=logs
            self.writer = tf.summary.FileWriter("./logs", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []

    def _build_net(self):
        # ------------------ all inputs ------------------------

        self.s = tf.placeholder(tf.float32, [None] + self.state_size + [4], name='s')  # input State
        self.s_ = tf.placeholder(tf.float32, [None] + self.state_size + [4], name='s_')  # input Next State
        self.r = tf.placeholder(tf.float32, [None, ], name='r')  # input Reward
        self.a = tf.placeholder(tf.int32, [None, ], name='a')  # input Action

        with tf.variable_scope('eval_net'):
            self.q_eval = tf.identity(self.network(self.s))

        with tf.variable_scope('target_net'):
            self.q_next = tf.identity(self.target_network(self.s_))

        with tf.variable_scope('q_target'):
            q_target = self.r + self.gamma * tf.reduce_max(self.q_next, axis=1, name='Qmax_s_') # shape=(None, )
            self.q_target = tf.stop_gradient(q_target)
        tf.summary.scalar('Q_mean', tf.reduce_mean(q_target))
        with tf.variable_scope('q_eval'):
            a_indices = tf.stack([tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a], axis=1) # shape=(action_dim, 2)
            self.q_eval_wrt_a = tf.gather_nd(params=self.q_eval, indices=a_indices)    # output Q(s,a)
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval_wrt_a, name='TD_error'))
        tf.summary.scalar('loss', self.loss)
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

    def store(self, state, action, reward, next_state, is_terminal, info):
        # rewards clipping
        if self.reward_clipping > 0.0:
            reward = np.clip(reward, -self.reward_clipping, self.reward_clipping)

        self.experience_replay.store(state, action, reward, next_state, is_terminal, info)

        if self.experience_replay.current_index >= 160:
            self.learn()

    def choose_action(self, observation):
        # to have batch dimension when feed into tf placeholder
        observation = observation[np.newaxis, :]

        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.target_replace_op)
            print('\ntarget_params_replaced\n')

        mini_batch = self.experience_replay.sample()
        if len(mini_batch) == 0:
            return 0

        batch_s = np.asarray([d[0] for d in mini_batch])
        batch_a = np.asarray( [d[1] for d in mini_batch] )
        batch_r = np.asarray( [d[2] for d in mini_batch] )
        batch_s_ = np.asarray( [d[3] for d in mini_batch])

        # states_mask = np.asarray( [not d[4] for d in minibatch] )
        feed_dict = {
            self.s: batch_s,
            self.a: batch_a,
            self.r: batch_r,
            self.s_: batch_s_,
        }
        _, cost = self.sess.run([self._train_op, self.loss],feed_dict=feed_dict)

        self.cost_his.append(cost)

        summary = self.sess.run(self.summary_op, feed_dict=feed_dict)
        self.writer.add_summary(summary, self.learn_step_counter)

        self.writer.flush()
        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1
