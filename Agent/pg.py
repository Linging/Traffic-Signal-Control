import numpy as np
import tensorflow as tf
from Agent.networks import *


np.random.seed(1)
tf.set_random_seed(1)


class PolicyGradient:
    def __init__(
            self,
            n_actions,
            learning_rate=0.02,
            reward_decay=0.95,
    ):
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = reward_decay

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []
        self.network = QNetworkNIPS([84,84],n_actions,'Net')
        self._build_net()

        self.sess = tf.Session()
        self.merged = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter("./pg_log", graph=tf.get_default_graph())
        self.sess.run(tf.global_variables_initializer())

    def weight_variable(self, shape):
        initializer = tf.contrib.layers.xavier_initializer()
        var = tf.Variable(initializer(shape))
        return  var

    def _build_net(self):
        self.tf_obs = tf.placeholder(tf.float32, [None, 84, 84, 4])
        self.tf_acts = tf.placeholder(tf.int32, [None, ], name="actions_num")
        self.tf_vt = tf.placeholder(tf.float32, [None, ], name="actions_value")

        all_act = tf.identity(self.network(self.tf_obs))

        self.all_act_prob = tf.nn.softmax(all_act, name='act_prob')  # use softmax to convert to probability

        with tf.name_scope('loss'):
            # to maximize total reward (log_p * R) is to minimize -(log_p * R), and the tf only have minimize(loss)
            neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=all_act, labels=self.tf_acts)   # this is negative log of chosen action
            # or in this way:
            # neg_log_prob = tf.reduce_sum(-tf.log(self.all_act_prob)*tf.one_hot(self.tf_acts, self.n_actions), axis=1)
            loss = tf.reduce_mean(neg_log_prob * self.tf_vt)  # reward guided loss

        tf.summary.scalar('loss', loss)

        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)

    def choose_action(self, observation):
        prob_weights = self.sess.run(self.all_act_prob, feed_dict={self.tf_obs: [observation]})
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())
        # select action w.r.t the actions prob
        return action

    def choose_action_test(self, observation):
        prob_weights = self.sess.run(self.all_act_prob, feed_dict={self.tf_obs: [observation]})
        action = np.argmax(prob_weights.shape[1])
        return action

    def store_transition(self, s, a, r):
        self.ep_obs.append([s])
        self.ep_as.append(a)
        self.ep_rs.append(r)

    def learn(self, ep):
        # discount and normalize episode reward
        discounted_ep_rs_norm = self._discount_and_norm_rewards()

        # train on episode
        train,summary = self.sess.run([self.train_op, self.merged], feed_dict={
             self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
             self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
             self.tf_vt: discounted_ep_rs_norm,  # shape=[None, ]
        })
        self.writer.add_summary(summary, ep)
        self.ep_obs, self.ep_as, self.ep_rs = [], [], []    # empty episode data
        return discounted_ep_rs_norm

    def _discount_and_norm_rewards(self):
        # discount episode rewards
        discounted_ep_rs = np.zeros_like(self.ep_rs)
        running_add = 0
        for t in reversed(range(0, len(self.ep_rs))):
            running_add = running_add * self.gamma + self.ep_rs[t]
            discounted_ep_rs[t] = running_add

        # normalize episode rewards
        discounted_ep_rs -= np.mean(discounted_ep_rs)
        discounted_ep_rs /= np.std(discounted_ep_rs)
        return discounted_ep_rs
