import numpy as np
import torch
from torch.optim import Adam
import gym
import time
import spinup.algos.pytorch.dqn.core as core
from spinup.utils.logx import EpochLogger
# from spinup.utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
# from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs

# from torch.utils.tensorboard import SummaryWriter 
# write = SummaryWriter("/home/masa/learning/rl/undergraduate/cjy/robot_design_sythesis/test/algo/tb/")


import gc
import logging
LOGGING_PATH = "/home/masa/learning/rl/undergraduate/cjy/robot_design_sythesis/test/algo/check.log"
logging.basicConfig(format='%(asctime)s-%(levelname)s: %(message)s', filename=LOGGING_PATH, level=logging.INFO)

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
      obs1=self.obs1_buf[idxs],
      obs2=self.obs2_buf[idxs],
      acts=self.acts_buf[idxs],
      rews=self.rews_buf[idxs],
      done=self.done_buf[idxs],
    )

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
  def get_action(eps, A, T, O, action_space=np.array([0, 1])):
    action = None

    if np.random.random() < eps:
      action = np.choose(np.random.randint(len(action_space)), action_space)

    else:
      action = network.act(A, T, O)

    if action in action_space:
      return action
    else: 
      return np.choose(np.random.randint(len(action_space)), action_space)

  # 加入动作掩码
  def get_action_with_mask(eps, A, T, O, action_space):
    from robotConfigDesign.envs.utils import generate_action_mask
    action_mask = generate_action_mask(A[0], env.action_space.n, action_space)
    action = None

    if np.random.random() < eps:
      valid_actions = np.where(action_mask == 0)[0]
      action = np.random.choice(valid_actions)
    else:
      # 把torch tensor转numpy
      masked_q_value = network.predict(A, T, O).numpy() + action_mask
      action = np.argmax(masked_q_value)

    return action

  # 测试模型
  def test_agent(n=2):
    for j in range(n):
      o, r, d, ep_ret, ep_len = test_env.reset(), 0, False, 0, 0
      _ = {'action_space': np.array([0, 1])}
      while not(d or (ep_len == max_ep_len)):
        o, r, d, _ = test_env.step(get_action_with_mask(0, o['A'], o['T'], o['O'], _['action_space']))
        # o, r, d, _ = test_env.step(get_action(0, o['A'], o['T'], o['O'], _['action_space']))
        ep_ret += r
        ep_len += 1
      logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)
      # write.add_scalar('TestEpRet', ep_ret, j)
      # write.add_scalar('TestEpLen', ep_len, j)

  # 开始训练
  def update(t):
    from robotConfigDesign.envs.utils import generate_action_mask
      
    batch = replay_buffer.sample_batch(batch_size=batch_size)
    states = batch['obs']
    next_states = batch['nobs']
    actions = torch.tensor(batch['acts'], dtype=torch.long)
    rewards = torch.Tensor(batch['rews'])
    dones = torch.tensor(batch['done'], dtype=torch.bool)

    with torch.no_grad():
      q_next = network.main(states['A'], states['T'], states['O'])
      # 根据states['O']是状态采样集合，生成对应长度的动作掩码，generate_action_mask仅针对单个状态
      next_q_values = network.target(next_states['A'], next_states['T'], next_states['O'])
      # print([a for a in next_states['A']])
      action_mask = torch.Tensor([generate_action_mask(a, env.action_space.n) for a in next_states['A']])
      masked_next_q_values = next_q_values + action_mask

      q_next[range(len(q_next)), actions] = (
        rewards + ~dones * gamma * 
        torch.max(masked_next_q_values, dim=1).values
      )
      # q_next[range(len(q_next)), actions] = (
      #   rewards + ~dones * gamma * 
      #   torch.max(network.target(next_states['A'], next_states['T'], next_states['O']), dim=1).values
      # )
    q_next = q_next.gather(1, actions.reshape(batch_size, 1))

    q_pred = network.main(states['A'], states['T'], states['O']).gather(1, actions.reshape(batch_size,1))
    for i in range(len(q_pred)):
      logger.store(QVals=q_pred[i].item())
    loss = network.crite(q_next, q_pred)
    network.optim.zero_grad()
    loss.backward()
    network.optim.step()

    logger.store(LossQ=loss.item())
    # write.add_scalar('LossQ', loss.item(), t)

  start_time = time.time()
  # env.render()
  o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
  _ = {'action_space': np.array([0, 1])}
  total_steps = steps_per_epoch * epochs

  # 主循环：在环境中收集经验，并在每个epoch中更新/记录
  for t in range(total_steps):
    epsilon = epsilon_start - (t * epsilon_decay)
    if epsilon < epsilon_end:
      epsilon = epsilon_end
    logger.store(Epsilon=epsilon)
    # write.add_scalar('Epsilon', epsilon, t)
    a = get_action_with_mask(epsilon, o['A'], o['T'], o['O'], _['action_space'])
    # a = get_action(epsilon, o['A'], o['T'], o['O'], _['action_space'])

    o2, r, d, _ = env.step(a)
    ep_ret += r
    ep_len += 1

    replay_buffer.store(o, a, r, o2, d)
    o = o2

    if d or (ep_len == max_ep_len):
      logger.store(EpRet=ep_ret, EpLen=ep_len)
      o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
      _ = {'action_space': np.array([0, 1])}

    if t % update_freq == 0:
      network.target_update()

    if t >= start_steps:
      update(t)
      
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
  from robotConfigDesign.envs import RobotConfigDesignEnv
  import os

  dir = '/home/masa/learning/rl/undergraduate/cjy/robot_design_sythesis/test/algo'

  envName = 'RobotConfigDesign-v0'
  env_fn = lambda : gym.make(envName)

  ac_kwargs = dict(hidden_sizes=[64, 64, 64], activation=torch.nn.LeakyReLU)

  logger_kwargs = dict(output_dir=dir + '/data/' + envName[:-3] + '-v9', exp_name=envName[:-3])

  dqn_2015(
    env_fn=env_fn, 
    ac_kwargs=ac_kwargs, 
    seed=0,
    steps_per_epoch=150,
    epochs=5000, 
    replay_size=int(1e6),
    gamma=1,
    epsilon_start=1,
    epsilon_decay=1e-5,
    epsilon_end=0.1,
    q_lr=1e-5,
    batch_size=int(32),
    start_steps=1500,
    max_ep_len=15,
    logger_kwargs=logger_kwargs,
    update_freq=150,
    save_freq=int(1)
  )