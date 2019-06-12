"""
Policy Gradient, Reinforcement Learning.
The cart pole example
View more on my tutorial page: https://morvanzhou.github.io/tutorials/
Using:
Tensorflow: 1.0
gym: 0.8.0
"""

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