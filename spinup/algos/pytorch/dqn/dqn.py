import numpy as np
import torch
from torch.optim import Adam
import gym
import time
import spinup.algos.pytorch.dqn.core as core
from spinup.utils.logx import EpochLogger
# from spinup.utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
# from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs

class ReplayBuffer_2015:
  def __init__(self, obs_dim, act_dim, size) -> None:
    self._obs = {i:o for i, o in obs_dim.sample().items()}
    self.obs_buf = dict()
    self.obs_buf['A'] = np.zeros(core.combined_shape(size, self._obs['A'].shape), dtype=np.float)
    self.obs_buf['T'] = np.zeros(core.combined_shape(size, self._obs['T'].shape), dtype=np.float)
    self.obs_buf['O'] = np.zeros(core.combined_shape(size, self._obs['O'].shape), dtype=np.float)
    self.nobs_buf = dict()
    self.nobs_buf['A'] = np.zeros(core.combined_shape(size, self._obs['A'].shape), dtype=np.float)
    self.nobs_buf['T'] = np.zeros(core.combined_shape(size, self._obs['T'].shape), dtype=np.float)
    self.nobs_buf['O'] = np.zeros(core.combined_shape(size, self._obs['O'].shape), dtype=np.float)

    self.acts_buf = np.zeros(
      core.combined_shape(size, act_dim), 
      dtype=np.long
    )

    self.rews_buf = np.zeros(size, dtype=np.float)
    self.done_buf = np.zeros(size, dtype=np.bool)
    self.ptr, self.size, self.max_size = 0, 0, size

  def store(self, obs, act, rew, next_obs, done:bool):
    self.obs_buf['A'][self.ptr] = np.array(obs['A'][0])
    self.obs_buf['T'][self.ptr] = np.array(obs['T'][0])
    self.obs_buf['O'][self.ptr] = np.array(obs['O'][0])
    self.nobs_buf['A'][self.ptr] = np.array(next_obs['A'][0])
    self.nobs_buf['T'][self.ptr] = np.array(next_obs['T'][0])
    self.nobs_buf['O'][self.ptr] = np.array(next_obs['O'][0])

    self.acts_buf[self.ptr] = act
    self.rews_buf[self.ptr] = rew
    self.done_buf[self.ptr] = done
    self.ptr = (self.ptr + 1) % self.max_size
    self.size = min(self.size + 1, self.max_size)

  def sample_batch(self, batch_size=32):
    idxs = np.random.randint(0, self.size, size=batch_size)
    return dict(
      obs={
        'A':self.obs_buf['A'][idxs],
        'T':self.obs_buf['T'][idxs],
        'O':self.obs_buf['O'][idxs],
      },
      nobs={
        'A':self.nobs_buf['A'][idxs],
        'T':self.nobs_buf['T'][idxs],
        'O':self.nobs_buf['O'][idxs],
      },
      acts=self.acts_buf[idxs],
      rews=self.rews_buf[idxs],
      done=self.done_buf[idxs],
    )

class ReplayBuffer:
  def __init__(self, obs_dim, act_dim, size) -> None:
    # self.obs1_buf = [[np.zeros(16), np.zeros(3), np.zeros(( 8, 8, 4 ))] for _ in range(size)]
    # self.obs2_buf = [[np.zeros(16), np.zeros(3), np.zeros((8, 8, 4))] for _ in range(size)]
    # self.obs2_buf = [None for _ in range(size)]
    self.obs1_buf = np.zeros(
      core.combined_shape(size, obs_dim), 
      dtype=np.float
    )
    self.obs2_buf = np.zeros(
      core.combined_shape(size, obs_dim), 
      dtype=np.float
    )
    self.acts_buf = np.zeros(
      core.combined_shape(size, act_dim), 
      dtype=np.long
    )
    self.rews_buf = np.zeros(size, dtype=np.float)
    self.done_buf = np.zeros(size, dtype=np.bool)
    self.ptr, self.size, self.max_size = 0, 0, size

  def store(self, obs, act, rew, next_obs, done:bool):
    # self.obs1_buf[self.ptr][0] = np.array(obs[0])
    # self.obs1_buf[self.ptr][1] = np.array(obs[1])
    # self.obs1_buf[self.ptr][2] = np.array(obs[2])
    # self.obs2_buf[self.ptr][0] = np.array(next_obs[0])
    # self.obs2_buf[self.ptr][1] = np.array(next_obs[1])
    # self.obs2_buf[self.ptr][2] = np.array(next_obs[2])
    # self.obs2_buf[self.ptr] = np.array(next_obs)
    self.obs1_buf[self.ptr] = obs
    self.obs2_buf[self.ptr] = next_obs
    self.acts_buf[self.ptr] = act
    self.rews_buf[self.ptr] = rew
    self.done_buf[self.ptr] = done
    self.ptr = (self.ptr + 1) % self.max_size
    self.size = min(self.size + 1, self.max_size)

  def sample_batch(self, batch_size=32):
    idxs = np.random.randint(0, self.size, size=batch_size)
    return dict(
      # obs1=np.array([ self.obs1_buf[i] for i in idxs ]),
      # obs2=np.array( [ self.obs2_buf[i] for i in idxs ] ),
      obs1=self.obs1_buf[idxs],
      obs2=self.obs2_buf[idxs],
      acts=self.acts_buf[idxs],
      rews=self.rews_buf[idxs],
      done=self.done_buf[idxs],
    )

def dqn_simple(
  env_fn,
  network=core.DQN_New_Double,
  ac_kwargs:dict=dict(),
  seed:int=0,
  steps_per_epoch:int=500,
  epochs:int=100,
  replay_size:int=int(1e6),
  gamma:float=0.99,
  epsilon_start:float=1,
  epsilon_decay:float=1e-4,
  epsilon_end:float=0.1,
  q_lr:float=1e-3,
  batch_size=100,
  start_steps=5000,
  max_ep_len:int=1000,
  logger_kwargs:dict=dict(),
  update_freq:int=100,
  save_freq:int=1
):
  """
  # Deep Q Network or Deep Q Learning (2015 版)

  ## Args:

  ### env_fn
    创建环境函数
    - 环境必须遵守OpenAI gym API

  ### network
    近似给定状态下所有可能Q值得torch网络类

  ### ac_kwargs
    网络结构描述字典

  ### seed (int)
    随机数种子

  ### steps_per_epoch (int)
    一次交互有多少个step

  ### epochs (int)
    一共有多少个epoch（开始交互到完成steps_per_epoch数量得交互）

  ### gamma (float)
    折扣因子
    - 超参数，总是在0到1之间

  ### epsilon_start (float)
    epsilon的初始值

  ### epsilon_decay (float)
    epsilon的衰减率

  ### epsilon_end (float) 
    epsilon的最小值

  ### q_lr (float) 
    学习率

  ### batch_size (int)
    一次训练的batch大小

  ### start_steps (int)
    开始训练前的随机交互步数

  ### max_ep_len (int)
    一个episode的最大长度

  ### logger_kwargs (dict): Keyword args for EpochLogger.
    日志记录参数

  ### update_freq (int)
    同步目标网络和主网络参数的频率（以step之间的间隔为单位）

  ### save_freq (int)
    保存当前策略和价值函数的频率（以epoch之间的间隔为单位）

  """
  # 初始化日志记录器
  logger = EpochLogger(**logger_kwargs)
  logger.save_config(locals())

  # 随机数种子
  torch.manual_seed(seed=seed)
  np.random.seed(seed=seed)

  # 实例化环境
  env, test_env = env_fn(), env_fn()
  obs_dim = env.observation_space.shape # (4,) for cartpoleEnv 
  act_dim = env.action_space.shape # () for cartpoleEnv
  act_num = env.action_space.n # 2 for cartpoleEnv

  ac_kwargs['action_space'] = env.action_space

  # 创建主网络和目标网络
  network = network(obs_dim[0], act_num, ac_kwargs['hidden_sizes'], q_lr, ac_kwargs['activation'])

  # 计算网络参数数量
  var_counts_main = core.count_vars(network.main)
  var_counts_target = core.count_vars(network.target)
  logger.log('\nNumber of parameters: \t main: {0}, \t target: {1}\n'.format(var_counts_main, var_counts_target))

  # 初始化经验池
  replay_buffer = ReplayBuffer(
    obs_dim=obs_dim,
    act_dim=act_dim,
    size=replay_size
  )

  # 初始化模型保存器
  logger.setup_pytorch_saver(network.main)

  # 基于epsilon-greedy策略选择动作
  def get_action(obs, eps):
    if np.random.random() < eps:
      return env.action_space.sample()
    return network.act(obs)

  # 测试函数
  def test_agent(n=10):
    for j in range(n):
      o, r, d, ep_ret, ep_len = test_env.reset(), 0, False, 0, 0
      while not(d or (ep_len == max_ep_len)):
        o, r, d, _ = test_env.step(get_action(o, 0))
        ep_ret += r
        ep_len += 1
      logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

  # 开始训练
  def update():
      
    batch = replay_buffer.sample_batch(batch_size=batch_size)
    states = batch['obs1']
    next_states = batch['obs2']
    actions = torch.tensor(batch['acts'], dtype=torch.long)
    rewards = torch.Tensor(batch['rews'])
    dones = torch.tensor(batch['done'], dtype=torch.bool)

    print(states['O'].shape)

    with torch.no_grad():
      q_next = network.main(torch.Tensor(states))
      q_next[range(len(q_next)), actions] = (
        # rewards + gamma * 
        rewards + ~dones * gamma * 
        torch.max(network.target(torch.Tensor(next_states)), dim=1).values
      )
    q_next = q_next.gather(1, actions.reshape(batch_size, 1))

    q_pred = network.main(torch.Tensor(states)).gather(1, actions.reshape(batch_size,1))
    for i in range(len(q_pred)):
      logger.store(QVals=q_pred[i].item())
    loss = network.crite(q_next, q_pred)
    network.optim.zero_grad()
    loss.backward()
    # torch.nn.utils.clip_grad_value_(network.main.parameters(), 100)
    network.optim.step()

    logger.store(LossQ=loss.item())
    # logger.store(LossQ=network.update(states, q_next))

  start_time = time.time()
  o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
  total_steps = steps_per_epoch * epochs

  # 主循环：在环境中收集经验，并在每个epoch中更新/记录
  for t in range(total_steps):
    epsilon = epsilon_start - (t * epsilon_decay)
    if epsilon < epsilon_end:
      epsilon = epsilon_end
    logger.store(Epsilon=epsilon)
    a = get_action(o, epsilon)

    o2, r, d, _ = env.step(a)
    ep_ret += r
    ep_len += 1

    replay_buffer.store(o, a, r, o2, d)
    d = False if ep_len == max_ep_len else d
    o = o2

    if d or (ep_len == max_ep_len):
      logger.store(EpRet=ep_ret, EpLen=ep_len)
      o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

    if t % update_freq == 0:
      network.target_update()

    if t >= start_steps:
      update()
      
    if t >= start_steps and (t - start_steps) % steps_per_epoch == 0:
      epoch = (t - start_steps) // steps_per_epoch
      if (epoch % save_freq == 0) or (epoch == epochs - 1):
        logger.save_state({'env': env}, None)

      test_agent()

      logger.log_tabular('Epoch', epoch)
      logger.log_tabular('EpRet', with_min_and_max=True)
      logger.log_tabular('TestEpRet', with_min_and_max=True)
      logger.log_tabular('EpLen', average_only=True)
      logger.log_tabular('TestEpLen', average_only=True)
      logger.log_tabular('TotalEnvInteracts', t)
      logger.log_tabular('QVals', with_min_and_max=True)
      logger.log_tabular('LossQ', average_only=True)
      logger.log_tabular('Epsilon', epsilon)
      logger.log_tabular('Time', time.time() - start_time)
      logger.dump_tabular()

def dqn_2015(
  env_fn,
  network=core.DQN_2015,
  ac_kwargs:dict=dict(),
  seed:int=0,
  steps_per_epoch:int=16,
  epochs:int=100,
  replay_size:int=int(1e6),
  gamma:float=0.99,
  epsilon_start:float=1,
  epsilon_decay:float=1e-4,
  epsilon_end:float=0.1,
  q_lr:float=1e-4,
  batch_size=100,
  start_steps=5000,
  max_ep_len:int=16,
  logger_kwargs:dict=dict(),
  update_freq:int=100,
  save_freq:int=1
):
  """
  # Deep Q Network or Deep Q Learning (2015 版)

  ## Args:

  ### env_fn
    创建环境函数
    - 环境必须遵守OpenAI gym API

  ### network
    近似给定状态下所有可能Q值得torch网络类

  ### ac_kwargs
    网络结构描述字典

  ### seed (int)
    随机数种子

  ### steps_per_epoch (int)
    一次交互有多少个step

  ### epochs (int)
    一共有多少个epoch（开始交互到完成steps_per_epoch数量得交互）

  ### gamma (float)
    折扣因子
    - 超参数，总是在0到1之间

  ### epsilon_start (float)
    epsilon的初始值

  ### epsilon_decay (float)
    epsilon的衰减率

  ### epsilon_end (float) 
    epsilon的最小值

  ### q_lr (float) 
    学习率

  ### batch_size (int)
    一次训练的batch大小

  ### start_steps (int)
    开始训练前的随机交互步数

  ### max_ep_len (int)
    一个episode的最大长度

  ### logger_kwargs (dict): Keyword args for EpochLogger.
    日志记录参数

  ### update_freq (int)
    同步目标网络和主网络参数的频率（以step之间的间隔为单位）

  ### save_freq (int)
    保存当前策略和价值函数的频率（以epoch之间的间隔为单位）

  """
  # 初始化日志记录器
  logger = EpochLogger(**logger_kwargs)
  logger.save_config(locals())

  # 随机数种子
  torch.manual_seed(seed=seed)
  np.random.seed(seed=seed)

  # 实例化环境
  # env, test_env = env_fn(), env_fn()
  env = env_fn()
  test_env = env
  obs_dim = env.observation_space # (4,) for cartpoleEnv 
  act_dim = env.action_space.shape # () for cartpoleEnv

  ac_kwargs['action_space'] = env.action_space

  # 创建主网络和目标网络
  network = network(env.observation_space, env.action_space, ac_kwargs['hidden_sizes'], q_lr, ac_kwargs['activation'])

  # 计算网络参数数量
  var_counts_main = core.count_vars(network.main)
  var_counts_target = core.count_vars(network.target)
  logger.log('\nNumber of parameters: \t main: {0}, \t target: {1}\n'.format(var_counts_main, var_counts_target))

  # 初始化经验池
  replay_buffer = ReplayBuffer_2015(
    obs_dim=obs_dim,
    act_dim=act_dim,
    size=replay_size
  )

  # 初始化模型保存器
  logger.setup_pytorch_saver(network.main)

  # 基于epsilon-greedy策略选择动作
  def get_action(eps, A, T, O):
    if np.random.random() < eps:
      return env.action_space.sample()
    return network.act(A, T, O)

  # 测试函数
  def test_agent(n=10):
    for j in range(n):
      o, r, d, ep_ret, ep_len = test_env.reset(), 0, False, 0, 0
      while not(d or (ep_len == max_ep_len)):
        o, r, d, _ = test_env.step(get_action(0, o['A'], o['T'], o['O']))
        ep_ret += r
        ep_len += 1
      logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

  # 开始训练
  def update():
      
    batch = replay_buffer.sample_batch(batch_size=batch_size)
    states = batch['obs']
    next_states = batch['nobs']
    actions = torch.tensor(batch['acts'], dtype=torch.long)
    rewards = torch.Tensor(batch['rews'])
    dones = torch.tensor(batch['done'], dtype=torch.bool)
    # 取出(batch_size, 3)的第一行
    # states = states.transpose()
    # next_states = next_states.transpose()

    with torch.no_grad():
      q_next = network.main(states['A'], states['T'], states['O'])
      q_next[range(len(q_next)), actions] = (
        # rewards + gamma * 
        rewards + ~dones * gamma * 
        torch.max(network.target(next_states['A'], next_states['T'], next_states['O']), dim=1).values
      )
    q_next = q_next.gather(1, actions.reshape(batch_size, 1))

    q_pred = network.main(states['A'], states['T'], states['O']).gather(1, actions.reshape(batch_size,1))
    for i in range(len(q_pred)):
      logger.store(QVals=q_pred[i].item())
    loss = network.crite(q_next, q_pred)
    network.optim.zero_grad()
    loss.backward()
    # torch.nn.utils.clip_grad_value_(network.main.parameters(), 100)
    network.optim.step()

    logger.store(LossQ=loss.item())
    # logger.store(LossQ=network.update(states, q_next))

  start_time = time.time()
  env.render()
  o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
  total_steps = steps_per_epoch * epochs

  # 主循环：在环境中收集经验，并在每个epoch中更新/记录
  for t in range(total_steps):
    epsilon = epsilon_start - (t * epsilon_decay)
    if epsilon < epsilon_end:
      epsilon = epsilon_end
    logger.store(Epsilon=epsilon)
    a = get_action(epsilon, o['A'], o['T'], o['O'])

    o2, r, d, _ = env.step(a)
    ep_ret += r
    ep_len += 1

    replay_buffer.store(o, a, r, o2, d)
    d = False if ep_len == max_ep_len else d
    o = o2

    if d or (ep_len == max_ep_len):
      logger.store(EpRet=ep_ret, EpLen=ep_len)
      o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

    if t % update_freq == 0:
      network.target_update()

    if t >= start_steps:
      update()
      
    if t >= start_steps and (t - start_steps) % steps_per_epoch == 0:
      epoch = (t - start_steps) // steps_per_epoch
      if (epoch % save_freq == 0) or (epoch == epochs - 1):
        logger.save_state({'env': env}, None)

      test_agent()

      logger.log_tabular('Epoch', epoch)
      logger.log_tabular('EpRet', with_min_and_max=True)
      logger.log_tabular('TestEpRet', with_min_and_max=True)
      logger.log_tabular('EpLen', average_only=True)
      logger.log_tabular('TestEpLen', average_only=True)
      logger.log_tabular('TotalEnvInteracts', t)
      logger.log_tabular('QVals', with_min_and_max=True)
      logger.log_tabular('LossQ', average_only=True)
      logger.log_tabular('Epsilon', epsilon)
      logger.log_tabular('Time', time.time() - start_time)
      logger.dump_tabular()

if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('--env', type=str, default='CartPole-v1')
  parser.add_argument('--hid', type=int, default=64)
  parser.add_argument('--l', type=int, default=1)
  parser.add_argument('--gamma', type=float, default=0.99)
  parser.add_argument('--seed', '-s', type=int, default=0)
  parser.add_argument('--epochs', type=int, default=50)
  parser.add_argument('--exp_name', type=str, default='dqn')
  args = parser.parse_args()

  from spinup.utils.run_utils import setup_logger_kwargs
  logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

  dqn_simple(
    lambda : gym.make(args.env),
    network=core.mlp,
    ac_kwargs=dict(
      hidden_sizes=[args.hid] * args.l,
    ),
    gamma=args.gamma,
    seed=args.seed,
    epochs=args.epochs,
    logger_kwargs=logger_kwargs
  )