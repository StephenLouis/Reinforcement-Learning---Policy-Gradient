# Reinforcement-Learning---Policy-Gradient
the implementation of Policy Gradient  
**如果觉得不错，麻烦点颗星哦！**
# 【强化学习】Policy Gradient算法详解
博客：https://blog.csdn.net/Gilgame/article/details/91404427
## 1.算法思想
之前的文章已经介绍了Q-Learning的相关知识及其实例：[【强化学习】Q-Learning 迷宫算法案例](https://blog.csdn.net/Gilgame/article/details/90669032)
Q-Learning 是一个基于价值value的方法，通过计算动作得分来决策的，它在确定了价值函数的基础上采用某种策略（贪婪-epslion）的方法去选取动作。
不同于基于最优价值的算法，Policy Gradient算法更着眼于算法的长期回报。策略梯度根据目标函数的梯度方向去寻找最优策略。策略梯度算法中，整个回合结束之后才会进行学习，所以策略梯度算法对全局过程有更好的把握。

## 2.Cart Pole游戏介绍
##### 游戏规则
cart pole即车杆游戏，游戏如下，很简单，游戏里面有一个小车，上有竖着一根杆子。小车需要左右移动来保持杆子竖直。如果杆子倾斜的角度大于15°，那么游戏结束。小车也不能移动出一个范围（中间到两边各2.4个单位长度）。
##### action
+ 左移
+ 右移
##### state variables
+ position of the cart on the track
+ angle of the pole with the vertical
+ cart velocity
+ rate of change of the angle

分别表示车的位置，杆子的角度，车速，角度变化率

##### 游戏奖励
在gym的Cart Pole环境（env）里面，左移或者右移小车的action之后，env都会返回一个+1的reward。到达200个reward之后，游戏也会结束
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190611160926683.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0dpbGdhbWU=,size_16,color_FFFFFF,t_70)
## 3.算法实现
##### 3.1 算法描述如下：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190611141350392.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0dpbGdhbWU=,size_16,color_FFFFFF,t_70)
##### 3.2 定义主更新的循环
###### 3.2.1创建环境
在写循环之前，跟之前Q-Learning一样，我们首先得创建环境。不过这次我们不像上次那么自己定义环境了，手动编写环境是一件很耗时间的事。有这么一个库，Open gym，他给我们提供了很多很好的模拟环境，方便我们去在这些环境中尝试我们的算法。如下，在import gym后，几行代码就能很方便的创建环境。
```python3
import gym
from RL_brain import PolicyGradient
import matplotlib.pyplot as plt

DISPLAY_REWARD_THRESHOLD = 400  # 当回合总reward大于400的时候显示模拟窗口
RENDER = False  # 边训练边显示会拖慢训练速度，我们等程序先学习一段时间

env = gym.make('CartPole-v0')   # 创建 CardPole这个模拟
env.seed(1)     # 创建随机种子
env = env.unwrapped # 取消限制

print(env.action_space) #输出可用的动作
print(env.observation_space)    # 显示可用 state 的 observation
print(env.observation_space.high)  # 显示 observation 最高值
print(env.observation_space.low)   # 显示 observation 最低值
```
###### 3.2.1主循环
接下来定义主更新循环，之前的Q—Learning在回合的每一步都更新参数，但是在这里，计算机跑完一整个回合才更新一次，故需要把这回合的transition存储起来。

```python3
# 定义使用 Policy_gradient 的算法
RL = PolicyGradient(
    n_actions=env.action_space.n,
    n_features=env.observation_space.shape[0],
    learning_rate=0.02,
    reward_decay=0.99,
    # output_graph=True,
)

for i_episode in range(3000):

    observation = env.reset()   # 获取回合 i_episode 第一个 observation

    while True:
        if RENDER: env.render() # 刷新环境

        action = RL.choose_action(observation)  # 选行为

        observation_, reward, done, info = env.step(action) # 获取下一个state

        RL.store_transition(observation, action, reward)    # 存储这一回合的transition

        if done:    # 一个回合结束，开始更新参数
            ep_rs_sum = sum(RL.ep_rs)   # 统计每回合的reward

            if 'running_reward' not in globals():
                running_reward = ep_rs_sum
            else:
                running_reward = running_reward * 0.99 + ep_rs_sum * 0.01
            if running_reward > DISPLAY_REWARD_THRESHOLD: RENDER = True     # 判断是否开始模拟
            print("episode:", i_episode, "  reward:", int(running_reward))

            vt = RL.learn() # 学习参数，输出vt

            if i_episode == 0:  # 画图
                plt.plot(vt)
                plt.xlabel('episode steps')
                plt.ylabel('normalized state-action value')
                plt.show()
            break

        observation = observation_
```

##### 3.2 Policy Gradient代码结构
**PolicyGradient算法主结构**
```python
class PolicyGradient:
    # 初始化
    def __init__(self, n_actions, n_features, learning_rate=0.01, reward_decay=0.95, output_graph=False):

    # 建立 policy gradient 神经网络
    def _build_net(self):

    # 选行为
    def choose_action(self, observation):

    # 存储回合 transition
    def store_transition(self, s, a, r):

    # 学习更新参数
    def learn(self, s, a, r, s_):

    # 衰减回合的 reward
    def _discount_and_norm_rewards(self):
```
**初始化，定义参数**
&emsp;**self.ep_obs**,**self.ep_as**,**self.ep_rs**分别存储了当前episode的**状态**，**动作**和**奖励**。
```python
     self.n_actions = n_actions
     self.n_features = n_features
     self.lr = learning_rate
     self.gamma = reward_decay

     self.ep_obs, self.ep_as, self.ep_rs = [], [], []
```
**建立 policy gradient 神经网络**

```python3
    def _build_net(self):	# 定义了两层的神经网络
        with tf.name_scope('inputs'):	# 模型的输入包括三部分，分别是观察值，动作和奖励值
            self.tf_obs = tf.placeholder(tf.float32, [None, self.n_features], name="observations")
            self.tf_acts = tf.placeholder(tf.int32, [None, ], name="actions_num")
            self.tf_vt = tf.placeholder(tf.float32, [None, ], name="actions_value")
        # fc1
        layer = tf.layers.dense(
            inputs=self.tf_obs,
            units=10,	# 输出个数
            activation=tf.nn.tanh,  # 激活函数
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='fc1'
        )
        # fc2
        all_act = tf.layers.dense(
            inputs=layer,
            units=self.n_actions,
            activation=None,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='fc2'
        )

        self.all_act_prob = tf.nn.softmax(all_act, name='act_prob')  # 使用softmax激活函数将行为转为概率

        with tf.name_scope('loss'):
            # 最大化 总体 reward (log_p * R) 就是在最小化 -(log_p * R), 而 tf 的功能里只有最小化 loss
            neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=all_act, labels=self.tf_acts)
            # or in this way:
            # neg_log_prob = tf.reduce_sum(-tf.log(self.all_act_prob)*tf.one_hot(self.tf_acts, self.n_actions), axis=1)
            loss = tf.reduce_mean(neg_log_prob * self.tf_vt)  # (vt = 本reward + 衰减的未来reward) 引导参数的梯度下降

        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)
```
**选行为**
&emsp;如果对Q-Learning还有印象的话，Q-Learning是采用90%的贪婪策略，10%随机搜索。而这里，我们直接根据概率选取行为。
```python
    def choose_action(self, observation):
        # 算出所有行为的概率
        prob_weights = self.sess.run(self.all_act_prob, feed_dict={self.tf_obs: observation[np.newaxis, :]})
        # 根据概率选行为
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())
        return action
```
**存储回合**
&emsp;之前说过，policy gradient是在一个完整的episode结束后才开始训练的，因此，在一个episode结束前，我们要存储这个episode所有的经验，即状态，动作和奖励。

```python
    def store_transition(self, s, a, r):
        self.ep_obs.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)
```
**计算整个回合的奖励值**
&emsp;我们之前存储的奖励是当前状态s采取动作a获得的即时奖励，而当前状态s采取动作a所获得的真实奖励应该是即时奖励加上未来直到episode结束的奖励贴现和。
&emsp;换一种说法，当前的奖励很大，并不代表着这就一定是正确的方向。在围棋对抗中，机器人会只以最后的输赢为评判标准，如果最后赢了，则之前每一步的都是正确的，如果最后输了，则之前的每一步都是错误的。对于Policy Gradient来说，他追求的是整个回合的奖励总和。

```python
    def _discount_and_norm_rewards(self):
        # 统计一个回合的奖励
        discounted_ep_rs = np.zeros_like(self.ep_rs)
        running_add = 0
        for t in reversed(range(0, len(self.ep_rs))):
            running_add = running_add * self.gamma + self.ep_rs[t]
            discounted_ep_rs[t] = running_add

        # 标准化回合奖励值
        discounted_ep_rs -= np.mean(discounted_ep_rs)
        discounted_ep_rs /= np.std(discounted_ep_rs)
        return discounted_ep_rs
```
**学习**
&emsp;在定义好上面所有的部件之后，我们就可以编写模型训练函数了，这里需要注意的是，我们喂给模型的并不是我们存储的奖励值，而是在经过上一步计算的奖励贴现和。另外，我们需要在每一次训练之后清空我们的经验池。
```python3
    # 学习更新参数
    def learn(self):
        # 衰减, 并标准化这回合的 reward
        discounted_ep_rs_norm = self._discount_and_norm_rewards()

        # train on episode
        self.sess.run(self.train_op, feed_dict={
             self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
             self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
             self.tf_vt: discounted_ep_rs_norm,  # shape=[None, ]
        })

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []     # 清空回合 data
        return discounted_ep_rs_norm    # 返回这一回合的 state-action value
```

以上就是我对Policy Gradient 算法实现的理解，如果不对的地方欢迎指出。
