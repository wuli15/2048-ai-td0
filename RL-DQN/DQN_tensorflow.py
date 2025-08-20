import os
import random
import numpy as np
from collections import deque
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

REPLAY_BUFFER_SIZE = 100000
EPISODES=100000
GAMMA = 0.95
EPSILON = 0.2
LEARNING_RATE = 0.0005
UPDATE_STEP=100

class DQN:
    def __init__(self,width, height, action_size, model_file):
        self.width = width
        self.height = height
        self.action_size = action_size
        self.replay_buffer = deque(maxlen=REPLAY_BUFFER_SIZE)
        self.epsilon = EPSILON
        self.model_path = r"models\model.ckpt"
        self.model_file = model_file
        self.create_network()
        self.create_updating_method()
        self.session=tf.InteractiveSession()
        self.episode_count = 0
        self.global_step = 0
        self.progress_file = "training_progress.npy"
        self.epsilon = 0.5

        if os.path.exists(r'models\model.ckpt.index'):
            print("model exists,loading saved model\n")
            self.saver = tf.train.Saver()
            self.saver.restore(self.session, self.model_path)
            self.episode_count = np.load(self.progress_file)
            self.epsilon*=pow(0.9995,self.episode_count)
        else:
            print("no model found,creating a new one\n")
            self.session.run(tf.global_variables_initializer())
            self.saver = tf.train.Saver()

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.01, shape=shape)
        return tf.Variable(initial)

    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    def create_network(self):
        with tf.name_scope("input"):
            self.state_input = tf.placeholder(tf.float32, shape=[None, self.width, self.height, 1])

        # 主网络
        with tf.variable_scope("network"):
            # 第一层卷积: 3x3 kernel, 64个输出通道
            w_conv1 = self.weight_variable([3, 3, 1, 64])
            b_conv1 = self.bias_variable([64])
            h_conv1 = tf.nn.relu(self.conv2d(self.state_input, w_conv1) + b_conv1)

            # 第二层卷积: 3x3 kernel, 128个输出通道
            w_conv2 = self.weight_variable([3, 3, 64, 128])
            b_conv2 = self.bias_variable([128])
            h_conv2 = tf.nn.relu(self.conv2d(h_conv1, w_conv2) + b_conv2)

            # 全局平均池化 (替代全连接层)
            h_global_pool = tf.reduce_mean(h_conv2, axis=[1, 2])  # 输出形状: [batch, 128]

            # 直接输出Q值 (128 -> 4)
            w_out = self.weight_variable([128, self.action_size])
            b_out = self.bias_variable([self.action_size])
            Q_value = tf.matmul(h_global_pool, w_out) + b_out
            self.Q_value = Q_value

        # 目标网络 (结构与主网络相同)
        with tf.variable_scope("target"):
            # 第一层卷积
            t_w_conv1 = self.weight_variable([3, 3, 1, 64])
            t_b_conv1 = self.bias_variable([64])
            t_h_conv1 = tf.nn.relu(self.conv2d(self.state_input, t_w_conv1) + t_b_conv1)

            # 第二层卷积
            t_w_conv2 = self.weight_variable([3, 3, 64, 128])
            t_b_conv2 = self.bias_variable([128])
            t_h_conv2 = tf.nn.relu(self.conv2d(t_h_conv1, t_w_conv2) + t_b_conv2)

            # 全局平均池化
            t_h_global_pool = tf.reduce_mean(t_h_conv2, axis=[1, 2])

            # 输出层
            t_w_out = self.weight_variable([128, self.action_size])
            t_b_out = self.bias_variable([self.action_size])
            t_Q_value = tf.matmul(t_h_global_pool, t_w_out) + t_b_out
            self.t_Q_value = t_Q_value

        # 目标网络参数更新操作
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='network')
        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target')
        with tf.variable_scope('soft_replacement'):
            self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

    def create_updating_method(self):
        self.action_input = tf.placeholder('float', [None,self.action_size])
        self.y_input = tf.placeholder('float', [None])
        Q_action=tf.reduce_sum(tf.multiply(self.Q_value,self.action_input),1)
        self.cost=tf.reduce_mean(tf.square(Q_action-self.y_input))

        tf.summary.scalar('loss', self.cost)

        with tf.name_scope('train_loss'):
            self.optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(self.cost)

    def choose_action(self, state):
        Q_value=self.Q_value.eval(feed_dict={self.state_input:[state]})[0]

        if random.random() <= self.epsilon:
            return random.randint(0,self.action_size-1)
        else:
            return np.argmax(Q_value)

    #used for test
    def act(self,state):
        pass

    def train(self, BATCH_SIZE):
        self.global_step+=1
        minibatch=random.sample(self.replay_buffer,BATCH_SIZE)

        state_batch=[dta[0] for dta in minibatch]
        action_batch=[dta[1] for dta in minibatch]
        reward_batch=[dta[2] for dta in minibatch]
        next_state_batch=[dta[3] for dta in minibatch]

        y_batch=[]
        q_value_batch=self.t_Q_value.eval(feed_dict={self.state_input:next_state_batch})

        for i in range(BATCH_SIZE):
            done=minibatch[i][4]
            if done:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(reward_batch[i] + GAMMA * np.max(q_value_batch[i]))

        self.optimizer.run(feed_dict={
            self.y_input: y_batch,
            self.action_input: action_batch,
            self.state_input: state_batch,
        })

        if len(self.replay_buffer) > 5e4 and self.global_step % UPDATE_STEP == 0:
            self.update()

    def store(self,state,action,reward,next_state,done):
        one_hot=np.zeros(self.action_size)
        one_hot[action]=1
        self.replay_buffer.append((state,one_hot,reward,next_state,done))
        if len(self.replay_buffer)>REPLAY_BUFFER_SIZE:
            self.replay_buffer.popleft()

    def update(self):
        self.session.run(self.target_replace_op)

    def save_model(self):
        self.save_path=self.saver.save(self.session,self.model_path)
        np.save('training_progress.npy',self.episode_count)
        print("Model saved in path: %s" % self.save_path)