import numpy as np
from gym.spaces import Box, Discrete
import gym

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.rmsprop import RMSprop
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical

from copy import deepcopy


def combined_shape(length: int, shape=None):
    '''
    合并数组维度，用于生成经验数组。

    参数
    --------
    length: int
      经验数组长度
    shape: None or int or a list of int
      观测值的维度

    返回值 
    ---
    newShape: tuple
      用于初始化经验数组维度的元组

    使用方法 
    ---
    >>> combined_shape(10, 4)
    (10, 4)
    >>> combined_shape(10, (2, 4))
    (10, 2, 4)
    >>> combined_shape(10, None)
    (10, )
    >>> np.zeros(combined_shape(buf_size, obs_dim), dtype=float32)

    '''
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def mlp(
    sizes: list,
    activation=nn.ReLU,
    output_activation=nn.Identity
):
    '''
    构造多层感知机

    参数 
    ---
    sizes: list
      网络结构
    activation: 激活函数
      插入到每层之后
    output_activation: softmax or Identity 
      分类使用softmax，回归用Identity

    返回值 
    ---
    torch.nn.Sequential()

    使用方法 
    ---
    >>> mlp([obs_size] + list(hidden_sizes) + [action_size], torch.nn.ReLU)

    '''
    layers = list()
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


def count_vars(module):
    '''
    计算网络权重参数的数量

    参数 
    ---
    module: torch.nn.module 的实例

    返回值 
    ---
    权重参数个数
    '''
    return sum([np.prod(p.shape) for p in module.parameters()])


def copy_operation(scope1, scope2):
    pass


class QNet(nn.Module):
    '''
    ## 网络类
    - 实现了网络的前向传播和选择动作的函数
    '''

    # 定义Net的一系列属性
    def __init__(self, observation_space, action_space, hidden_sizes, actiavtion=nn.ReLU):
        super(QNet, self).__init__()
        self._obs = {i: o for i, o in observation_space.sample().items()}
        self._fc_A = nn.Linear(self._obs['A'].size, 64)
        self._fc_T = nn.Linear(self._obs['T'].size, 64)

        self._conv_O = nn.Conv3d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self._maxPool = nn.MaxPool3d(kernel_size=3, stride=3, padding=0)
  
        self._ob_n = 64 * 3
        
        self._ac_n = action_space.n if isinstance(
            action_space, gym.spaces.Discrete) else action_space.shape[0]
        self.model = mlp([self._ob_n] + list(hidden_sizes) +
                         [self._ac_n], activation=actiavtion)

    # 定义forward函数 (x为状态)
    def forward(self, A, T, O):
        A = F.relu(self._fc_A(torch.Tensor(A)))
        T = F.relu(self._fc_T(torch.Tensor(T)))
        O = F.relu(self._conv_O(torch.Tensor(O).unsqueeze(1)))
        O = self._maxPool(O)
        O = O.view(O.size(0), -1)
        # print(A.shape, T.shape, O.shape)
        x = torch.cat([A, T, O], dim=1)
        return self.model(torch.Tensor(x))

    def act(self, A, T, O):
        with torch.no_grad():
            return torch.argmax(self.forward(A, T, O)).item()


class DQN_2015:
    '''
    ## DQN-2015
    - 实现了DQN的主要功能(主网络更新，目标网络更新，选择动作)
    '''

    def __init__(self, observation_space, action_space, hidden_sizes, lr, actiavtion=nn.Tanh) -> None:
        self.main = QNet(observation_space, action_space,
                        hidden_sizes, actiavtion)
        self.target = deepcopy(self.main)
        # self.optim = torch.optim.SGD(self.main.parameters(), lr=lr)
        # self.optim = RMSprop(self.main.parameters(), lr=lr)
        self.optim = torch.optim.Adam(self.main.parameters(), lr=lr)
        self.crite = torch.nn.MSELoss()

    def target_update(self):
        self.target.load_state_dict(self.main.state_dict())

    def act(self, A, T, O):
        return self.main.act(A, T, O)


if __name__ == '__main__':
    from robotConfigDesign.envs import RobotConfigDesignEnv
    import gym
    env = gym.make('RobotConfigDesign-v0')
    print(env.observation_space.shape is None)
    print(env.action_space.shape is None)
    print(combined_shape(10, ()))
    print(combined_shape(10, None))

    qnet = QNet(env.observation_space, env.action_space, [64, 64])
