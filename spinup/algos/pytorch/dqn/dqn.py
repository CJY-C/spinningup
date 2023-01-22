import numpy as np
import torch
from torch.optim import Adam
import gym
import time
import spinup.algos.pytorch.dqn.core as core
from spinup.utils.logx import EpochLogger
# from spinup.utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
# from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs

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
  # Set up logger and save configuration 
  logger = EpochLogger(**logger_kwargs)
  logger.save_config(locals())

  # Random seed 
  # TODO: explain these code
  torch.manual_seed(seed=seed)
  np.random.seed(seed=seed)

  # Instantiate environment
  env, test_env = env_fn(), env_fn()
  obs_dim = env.observation_space.shape # (4,) for cartpoleEnv 
  act_dim = env.action_space.shape # () for cartpoleEnv
  act_num = env.action_space.n # 2 for cartpoleEnv

  ac_kwargs['action_space'] = env.action_space

  # Create nueral network module
  # TODO: implement the code below
  # network = core.DQN_MF()
  network = network(obs_dim[0], act_num, ac_kwargs['hidden_sizes'], q_lr, ac_kwargs['activation'])
  # network = network(list(obs_dim) + list(ac_kwargs['hidden_sizes']) + [act_num], torch.nn.ReLU)

  # count variables 
  # TODO: test this
  var_counts_main = core.count_vars(network.main)
  var_counts_target = core.count_vars(network.target)
  logger.log('\nNumber of parameters: \t main: {0}, \t target: {1}\n'.format(var_counts_main, var_counts_target))

  # Set up replay buffer
  replay_buffer = ReplayBuffer(
    obs_dim=obs_dim,
    act_dim=act_dim,
    size=replay_size
  )

  # Set up optimizers for Q function
  # TODO: explain code below
  # q_criterion = torch.nn.MSELoss()
  # q_optimizer = torch.optim.SGD(network.parameters(), lr=q_lr)
  # q_optimizer = Adam(network.main.parameters(), lr=q_lr)

  # Set up model saving
  # TODO: explain code below
  logger.setup_pytorch_saver(network.main)

  def get_action(obs, eps):
    if np.random.random() < eps:
      return env.action_space.sample()
    return network.act(obs)

  def test_agent(n=10):
    for j in range(n):
      o, r, d, ep_ret, ep_len = test_env.reset(), 0, False, 0, 0
      while not(d or (ep_len == max_ep_len)):
        o, r, d, _ = test_env.step(get_action(o, 0))
        ep_ret += r
        ep_len += 1
      logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

  # update function
  # TODO: implememt this 
  def update():
      
    batch = replay_buffer.sample_batch(batch_size=batch_size)
    states = torch.Tensor(batch['obs1'])
    next_states = torch.Tensor(np.array(batch['obs2']))
    actions = torch.tensor(batch['acts'], dtype=torch.long)
    rewards = torch.Tensor(batch['rews'])
    dones = torch.tensor(batch['done'], dtype=torch.bool)

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

  # Main loop: collect experience in env and update/log each epoch
  # TODO: implement this
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


def dqn_simpl(
  env_fn,
  network=core.DQN_New_Double,
  ac_kwargs:dict=dict(),
  seed:int=0,
  steps_per_epoch:int=1000,
  epochs:int=100,
  replay_size:int=int(1e3),
  gamma:float=0.99,
  epsilon_start:float=1,
  epsilon_decay:float=1e-4,
  epsilon_end:float=0.1,
  q_lr:float=1e-4,
  batch_size=100,
  start_steps=32,
  max_ep_len:int=1000,
  logger_kwargs:dict=dict(),
  update_frep:int=1000,
  save_freq:int=1
):
  """
  # Deep Q Network or Deep Q Learning (Simple version)

  ## Args:

  ### env_fn
    A function which creates a copy of the environment.
    - The environment must satisfy the OpenAI Gym API.

  ### network
    A torch module Sequential which approximates the Q value of all possible actions for a given state. 
    - fixed first

  ### seed (int)
    Seed for random number generators.

  ### steps_per_epoch (int)
    Number of steps of interaction (s-a pairs)

  ### epochs (int)
    Number of epochs of interaction
    - need details.

  ### gamma (float)
    Discount factor
    - Always between 0 and 1.

  ### lr (float) 
    Learning rate.

  ### max_ep_len (int)
    Maximum length of trajectory / episode / rollout.

  ### logger_kwargs (dict): Keyword args for EpochLogger.

  ### save_freq (int)
    How often (in the terms of gap between epochs) to save the current policy and value function.

  """

  # Set up logger and save configuration 
  # TODO: explain these code
  logger = EpochLogger(**logger_kwargs)
  logger.save_config(locals())

  # Random seed 
  # TODO: explain these code
  torch.manual_seed(seed=seed)
  np.random.seed(seed=seed)

  # Instantiate environment
  env, test_env = env_fn(), env_fn()
  obs_dim = env.observation_space.shape # (4,) for cartpoleEnv 
  act_dim = env.action_space.shape # () for cartpoleEnv
  act_num = env.action_space.n # 2 for cartpoleEnv

  ac_kwargs['action_space'] = env.action_space

  # Create nueral network module
  # TODO: implement the code below
  # network = core.DQN_MF()
  network = network(obs_dim[0], act_num, ac_kwargs['hidden_sizes'], q_lr, torch.nn.LeakyReLU)
  # network = network(list(obs_dim) + list(ac_kwargs['hidden_sizes']) + [act_num], torch.nn.ReLU)

  # count variables 
  # TODO: test this
  # var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.v])
  # var_counts_main = core.count_vars(network.main)
  # var_counts_target = core.count_vars(network.target)
  # logger.log('\nNumber of parameters: \t main: {0}, \t target: {1}\n'.format(var_counts_main, var_counts_target))

  # Set up replay buffer
  # TODO: implement ReplayBuffer and code below
  replay_buffer = ReplayBuffer(
    obs_dim=obs_dim,
    act_dim=act_dim,
    size=replay_size
  )

  # Set up optimizers for Q function
  # TODO: explain code below
  # q_criterion = torch.nn.MSELoss()
  # q_optimizer = torch.optim.SGD(network.parameters(), lr=q_lr)
  # q_optimizer = Adam(network.main.parameters(), lr=q_lr)

  # Set up model saving
  # TODO: explain code below
  logger.setup_pytorch_saver(network.main)

  def get_action(obs, eps):
    if np.random.random() < eps:
      a = env.action_space.sample()
    else:
      a = network.act(obs)
    return a

  def test_agent(n=10):
    for j in range(n):
      o, r, d, ep_ret, ep_len = test_env.reset(), 0, False, 0, 0
      while not(d or (ep_len == max_ep_len)):
        o, r, d, _ = test_env.step(get_action(o, 0))
        ep_ret += r
        ep_len += 1
      logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

  # update function
  # TODO: implememt this 
  def update():
    batch = replay_buffer.sample_batch(batch_size=batch_size)
    states = torch.Tensor(batch['obs1'])
    next_states = torch.Tensor(np.array(batch['obs2']))
    actions = torch.tensor(batch['acts'], dtype=torch.long)
    rewards = torch.Tensor(batch['rews'])
    dones = torch.tensor(batch['rews'], dtype=torch.int64)

    q_next = network.main.predict(states)
    q_next[range(len(q_next)), actions] = (
      rewards + ~dones * gamma * 
      torch.max(network.main.predict(next_states), dim=1).values
    )

    logger.store(LossQ=network.update(states, q_next))

  start_time = time.time()
  o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
  total_steps = steps_per_epoch * epochs

  # Main loop: collect experience in env and update/log each epoch
  # TODO: implement this
  for t in range(total_steps):
    epsilon = epsilon_start - (t * epsilon_decay)
    if epsilon < epsilon_end:
      epsilon = epsilon_end
    logger.store(Epsilon=epsilon)
    a = get_action(o, epsilon)

    o2, r, d, _ = env.step(a)
    ep_ret += r
    ep_len += 1

    d = False if ep_len == max_ep_len else d
    replay_buffer.store(o, a, r, o2, d)

    update()

    o = o2

    if d or (ep_len == max_ep_len):
      logger.store(EpRet=ep_ret, EpLen=ep_len)
      o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

    if (t + 1) % steps_per_epoch == 0:
      epoch = (t +1) // steps_per_epoch
    # replay_buffer.store(o, a, r, o2, d)
      if (epoch % save_freq == 0) or (epoch == epochs - 1):
        logger.save_state({'env': env}, None)

      test_agent()

      logger.log_tabular('Epoch', epoch)
      logger.log_tabular('EpRet', with_min_and_max=True)
      logger.log_tabular('TestEpRet', with_min_and_max=True)
      logger.log_tabular('EpLen', average_only=True)
      logger.log_tabular('TestEpLen', average_only=True)
      logger.log_tabular('TotalEnvInteracts', t)
      # logger.log_tabular('QVals', with_min_and_max=True)
      logger.log_tabular('LossQ', average_only=True)
      logger.log_tabular('Epsilon', epsilon)
      logger.log_tabular('Time', time.time() - start_time)
      logger.dump_tabular()

def dqn_sim(
  env_fn,
  network=core.DQN_orign,
  ac_kwargs:dict=dict(),
  seed:int=0,
  steps_per_epoch:int=1000,
  epochs:int=100,
  replay_size:int=int(1e5),
  gamma:float=0.99,
  epsilon_start:float=1,
  epsilon_decay:float=1e-4,
  epsilon_end:float=0.1,
  q_lr:float=1e-4,
  batch_size=100,
  start_steps=100,
  max_ep_len:int=1000,
  logger_kwargs:dict=dict(),
  update_frep:int=1000,
  save_freq:int=1
):
  """
  # Deep Q Network or Deep Q Learning (Simple version)

  ## Args:

  ### env_fn
    A function which creates a copy of the environment.
    - The environment must satisfy the OpenAI Gym API.

  ### network
    A torch module Sequential which approximates the Q value of all possible actions for a given state. 
    - fixed first

  ### seed (int)
    Seed for random number generators.

  ### steps_per_epoch (int)
    Number of steps of interaction (s-a pairs)

  ### epochs (int)
    Number of epochs of interaction
    - need details.

  ### gamma (float)
    Discount factor
    - Always between 0 and 1.

  ### lr (float) 
    Learning rate.

  ### max_ep_len (int)
    Maximum length of trajectory / episode / rollout.

  ### logger_kwargs (dict): Keyword args for EpochLogger.

  ### save_freq (int)
    How often (in the terms of gap between epochs) to save the current policy and value function.

  """

  # Set up logger and save configuration 
  # TODO: explain these code
  logger = EpochLogger(**logger_kwargs)
  logger.save_config(locals())

  # Random seed 
  # TODO: explain these code
  torch.manual_seed(seed=seed)
  np.random.seed(seed=seed)

  # Instantiate environment
  env, test_env = env_fn(), env_fn()
  obs_dim = env.observation_space.shape # (4,) for cartpoleEnv 
  act_dim = env.action_space.shape # () for cartpoleEnv
  act_num = env.action_space.n # 2 for cartpoleEnv

  ac_kwargs['action_space'] = env.action_space

  # Create nueral network module
  # TODO: implement the code below
  # network = core.DQN_MF()
  network = network(obs_dim[0], act_num, ac_kwargs['hidden_sizes'], q_lr, torch.nn.LeakyReLU)
  # network = network(list(obs_dim) + list(ac_kwargs['hidden_sizes']) + [act_num], torch.nn.ReLU)

  # count variables 
  # TODO: test this
  # var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.v])
  # var_counts_main = core.count_vars(network.main)
  # var_counts_target = core.count_vars(network.target)
  # logger.log('\nNumber of parameters: \t main: {0}, \t target: {1}\n'.format(var_counts_main, var_counts_target))

  # Set up replay buffer
  # TODO: implement ReplayBuffer and code below
  replay_buffer = ReplayBuffer(
    obs_dim=obs_dim,
    act_dim=act_dim,
    size=replay_size
  )

  # Set up optimizers for Q function
  # TODO: explain code below
  # q_criterion = torch.nn.MSELoss()
  # q_optimizer = torch.optim.SGD(network.parameters(), lr=q_lr)
  # q_optimizer = Adam(network.main.parameters(), lr=q_lr)

  # Set up model saving
  # TODO: explain code below
  logger.setup_pytorch_saver(network.main)

  def get_action(obs, eps):
    if np.random.random() < eps:
      a = env.action_space.sample()
    else:
      a = network.act(obs)
    return a

  def test_agent(n=10):
    for j in range(n):
      o, r, d, ep_ret, ep_len = test_env.reset(), 0, False, 0, 0
      while not(d or (ep_len == max_ep_len)):
        o, r, d, _ = test_env.step(get_action(o, 0))
        ep_ret += r
        ep_len += 1
      logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

  # update function
  # TODO: implememt this 
  def update():
    batch = replay_buffer.sample_batch(batch_size=batch_size)
    states = torch.Tensor(batch['obs1'])
    next_states = torch.Tensor(np.array(batch['obs2']))
    actions = torch.tensor(batch['acts'], dtype=torch.long)
    rewards = torch.Tensor(batch['rews'])
    dones = torch.tensor(batch['rews'], dtype=torch.int64)

    q_pred = network.main(states).gather(1, actions.reshape(batch_size, 1))
    q_target = (
      rewards + ~dones * gamma * 
      torch.max(network.main.predict(next_states), dim=1).values
    ).reshape(batch_size, 1)

    logger.store(LossQ=network.update(q_pred, q_target))

  start_time = time.time()
  o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
  total_steps = steps_per_epoch * epochs

  # Main loop: collect experience in env and update/log each epoch
  # TODO: implement this
  for t in range(total_steps):
    epsilon = epsilon_start - (t * epsilon_decay)
    if epsilon < epsilon_end:
      epsilon = epsilon_end
    logger.store(Epsilon=epsilon)
    a = get_action(o, epsilon)

    o2, r, d, _ = env.step(a)
    ep_ret += r
    ep_len += 1

    d = False if ep_len == max_ep_len else d
    replay_buffer.store(o, a, r, o2, d)

    if t >= start_steps:
      update()

    o = o2

    if d or (ep_len == max_ep_len):
      logger.store(EpRet=ep_ret, EpLen=ep_len)
      o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

    if (t + 1) % steps_per_epoch == 0:
      epoch = (t +1) // steps_per_epoch
    # replay_buffer.store(o, a, r, o2, d)
      if (epoch % save_freq == 0) or (epoch == epochs - 1):
        logger.save_state({'env': env}, None)

      test_agent()

      logger.log_tabular('Epoch', epoch)
      logger.log_tabular('EpRet', with_min_and_max=True)
      logger.log_tabular('TestEpRet', with_min_and_max=True)
      logger.log_tabular('EpLen', average_only=True)
      logger.log_tabular('TestEpLen', average_only=True)
      logger.log_tabular('TotalEnvInteracts', t)
      # logger.log_tabular('QVals', with_min_and_max=True)
      logger.log_tabular('LossQ', average_only=True)
      logger.log_tabular('Epsilon', epsilon)
      logger.log_tabular('Time', time.time() - start_time)
      logger.dump_tabular()


    # TODO: double network

    # if t > start_steps:

    #   if (t-1) % update_frep == 0:
    #     network.target_update()

    #   # batch = replay_buffer.sample_batch(batch_size=batch_size)
    #   # states = torch.Tensor(batch['obs1'])
    #   # next_states = torch.Tensor(np.array(batch['obs2']))
    #   # actions = torch.tensor(batch['acts'], dtype=torch.int64)
    #   # rewards = torch.Tensor(batch['rews'])
    #   # dones = torch.tensor(batch['rews'], dtype=torch.int64)

    #   # q_values = network.target_predict(states=states)
    #   # q_values[range(len(q_values)), actions] = (
    #   #   rewards + ~dones * gamma * 
    #   #   torch.max(network.target_predict(next_states), dim=1).values
    #   # )
    #   # logger.store(QVals=)
    #   update()

    # if t > start_steps and (t - start_steps) % steps_per_epoch == 0:
    #   epoch = (t - start_steps) // steps_per_epoch

    #   if (epoch % save_freq == 0) or (epoch == epochs - 1):
    #     logger.save_state({'env': env}, None)

    #   test_agent()

    #   logger.log_tabular('Epoch', epoch)
    #   logger.log_tabular('EpRet', with_min_and_max=True)
    #   logger.log_tabular('TestEpRet', with_min_and_max=True)
    #   logger.log_tabular('EpLen', average_only=True)
    #   logger.log_tabular('TestEpLen', average_only=True)
    #   logger.log_tabular('TotalEnvInteracts', t)
    #   # logger.log_tabular('QVals', with_min_and_max=True)
    #   logger.log_tabular('LossQ', average_only=True)
    #   logger.log_tabular('Epsilon', epsilon)
    #   logger.log_tabular('Time', time.time() - start_time)
    #   logger.dump_tabular()



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