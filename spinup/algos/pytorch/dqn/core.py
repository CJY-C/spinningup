import numpy as np
import scipy.signal
from gym.spaces import Box, Discrete

import torch
import torch.nn as nn
from torch.optim.rmsprop import RMSprop
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical

from copy import deepcopy

def combined_shape(length: int, shape=None):
    '''
    Using to generate data buffer

    Parameter
    --------
    length: int
      the length of data buffer
    shape: None or int or a list of int
      the shape of a sample 

    Return
    ---
    newShape: tuple
      for initialize the data buffer.

    Usage
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
    construct a torch version multilayer perceptron.

    Parameter
    ---
    sizes: list
      network structure
    activation: one of the activation functions of pytorch
      insert to the back of each Linear block
    output_activation: softmax or not

    Return
    ---
    torch.nn.Sequential()

    Usage
    ---
    >>> mlp([obs_size] + list(hidden_sizes) + [action_size], torch.nn.ReLU)

    '''
    layers = list()
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [ nn.Linear(sizes[j], sizes[j+1]), act() ]
    return nn.Sequential(*layers)

def count_vars(module):
  '''
  count the number of parameters in the neural network.

  Parameters
  ---
  module: instence of torch.nn.module

  Return
  ---
  val: total number of parameters
  '''
  return sum([np.prod(p.shape) for p in module.parameters()])

def copy_operation(scope1, scope2):
  pass


class DQN_Simple(nn.Module):
  def __init__(self, observation_space, action_space, hidden_sizes, actiavtion=nn.Tanh) -> None:
    super().__init__()
    # super(Network, self).__init__()
    self.model = mlp([observation_space] + list(hidden_sizes) + [action_space], activation=actiavtion)
  
  def forward(self, x):
    return self.model(torch.Tensor(x))

  def predict(self, states):
    with torch.no_grad():
      return self.model(torch.Tensor(states))
  
  def act(self, states):
    return torch.argmax(self.predict(states=states)).item()


class DQN_Double(nn.Module):
  def __init__(self, observation_space, action_space, hidden_sizes, actiavtion=nn.Tanh) -> None:
    super().__init__()
    self.main = mlp([observation_space] + list(hidden_sizes) + [action_space], activation=actiavtion)
    self.target = deepcopy(self.main)
  
  def forward(self, x):
    return self.main(torch.Tensor(x))

  def target_update(self):
    self.target.load_state_dict(self.main.state_dict())

  def target_predict(self, states):
    return self.target(torch.Tensor(states)).detach()
    # with torch.no_grad():
    #   return self.target(torch.Tensor(states))

  def predict(self, states):
    return self.main(torch.Tensor(states)).detach()
    # with torch.no_grad():
    #   return self.main(torch.Tensor(states))
    # self.target_predict(states)
  
  def act(self, states):
    return torch.argmax(self.target_predict(states=states)).item()

class DQN_MF():
  def __init__(self) -> None:
    self.main = Net()
    self.target = Net()
    # self.optim = torch.optim.Adam()
    # self.crite = torch.nn.MSELoss()
  
  def target_update(self):
    self.target.load_state_dict(self.main.state_dict())

  def target_predict(self, states):
    with torch.no_grad():
      return self.target(torch.Tensor(states))

  def predict(self, states):
    with torch.no_grad():
      return self.main(torch.Tensor(states))
    # self.target_predict(states)
  
  def act(self, states):
    return torch.argmax(self.predict(states=states)).item()

# 定义Net类 (定义网络)
class Net(nn.Module):
  def __init__(self, observation_space, action_space, hidden_sizes, actiavtion=nn.Tanh):                                                         # 定义Net的一系列属性
    # nn.Module的子类函数必须在构造函数中执行父类的构造函数
    super(Net, self).__init__()                                             # 等价与nn.Module.__init__()
    self.model = mlp([observation_space] + list(hidden_sizes) + [action_space], activation=actiavtion)

  def forward(self, x):                                                       # 定义forward函数 (x为状态)
    return self.model(torch.Tensor(x))

  def act(self, obs): 
    with torch.no_grad():
      return torch.argmax(self.model(torch.Tensor(obs))).item()

class DQN_orign:
  def __init__(self, observation_space, action_space, hidden_sizes, lr, actiavtion=nn.Tanh) -> None:
    self.main = Net(observation_space, action_space, hidden_sizes, actiavtion)
    # self.optim = torch.optim.SGD(self.main.parameters(), lr=lr)
    self.optim = torch.optim.Adam(self.main.parameters(), lr=lr)
    self.crite = torch.nn.SmoothL1Loss()

  def update(self, q_pred, q_target):
  # def update(self, obs, q_target):
    # q_pred = self.main(torch.Tensor(obs))
    loss = self.crite(q_target, q_pred)
    self.optim.zero_grad()
    loss.backward()
    for param in self.main.parameters():
        param.grad.data.clamp_(-1, 1)
    self.optim.step()
    return loss.item()

  def act(self, obs): 
    return self.main.act(obs)
  

class DQN_New_Double:
  def __init__(self, observation_space, action_space, hidden_sizes, lr, actiavtion=nn.Tanh) -> None:
    self.main = Net(observation_space, action_space, hidden_sizes, actiavtion)
    self.target = deepcopy(self.main)
    # self.optim = torch.optim.SGD(self.main.parameters(), lr=lr)
    self.optim = torch.optim.Adam(self.main.parameters(), lr=lr)
    # self.optim = RMSprop(self.main.parameters(), lr=lr)
    # self.crite = torch.nn.MSELoss()
    self.crite = ClipMSE()

  def update(self, obs, q_target):
    q_pred = self.main(torch.Tensor(obs))
    loss = self.crite(q_target, q_pred)
    self.optim.zero_grad()
    loss.backward()
    for param in self.main.parameters():
        param.grad.data.clamp_(-1, 1)
    self.optim.step()
    return loss.item()

  def target_update(self):
    self.target.load_state_dict(self.main.state_dict())
  
  def act(self, obs): 
    return self.main.act(obs)

# TODO: 自定义loss 加裁剪
class ClipMSE(nn.Module):
  def __init__(self) -> None:
    super().__init__()
  
  def forward(self, x, y):
    delta = x-y
    return torch.mean(torch.pow(delta, 2))