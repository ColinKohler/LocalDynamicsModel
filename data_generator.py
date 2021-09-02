import gc
import time
import numpy as np
import numpy.random as npr
import ray
import torch
import functools

from data.episode_history import EpisodeHistory
from agents.adn_agent import ADNAgent
from agents.shooting_agent import ShootingAgent
deom agents.dyna_agent import DynaAgent
from agents.fc_dqn_agent import FCDQNAgent
from agents.rot_fc_dqn_agent import RotFCDQNAgent
from data import data_utils
from data import constants
import utils

from helping_hands_rl_envs import env_factory

@ray.remote
class DataGenerator(object):
  def __init__(self, initial_checkpoint, config, seed):
    self.seed = seed
    self.config = config
    self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    env_type, env_config, _ = data_utils.getEnvConfig(self.config.env_type, self.config.use_rot)
    self.env = env_factory.createEnvs(0, 'pybullet', env_type, env_config)

    npr.seed(self.seed)
    torch.manual_seed(self.seed)

    if self.config.agent == 'adn':
      self.agent = ADNAgent(self.config, self.device)
    elif self.config.agent == 'shooting':
      self.agent = ShootingAgent(self.config, self.device)
    elif self.config.agent == 'dyna':
      self.agent = DynaAgent(self.config, self.device)
    elif self.config.agent == 'fc_dqn':
      self.agent = FCDQNAgent(self.config, self.device)
    elif self.config.agent == 'rot_fc_dqn':
      self.agent = RotFCDQNAgent(self.config, self.device)
    else:
      raise ValueError
    self.agent.setWeights(initial_checkpoint['weights'])

  def continuousDataGen(self, shared_storage, replay_buffer, test_mode=False):
    while (
      ray.get(shared_storage.getInfo.remote('training_step')) < self.config.training_steps and \
      not ray.get(shared_storage.getInfo.remote('terminate'))
    ):
      # NOTE: This might be a inificent but we can't check the training step really due to async
      self.agent.setWeights(ray.get(shared_storage.getInfo.remote('weights')))

      training_step = ray.get(shared_storage.getInfo.remote('training_step'))
      if self.config.agent == 'adn' and training_step % self.config.decay_action_sample_pen == 0 and training_step > 0:
        self.agent.decayActionSamplePen()

      if not test_mode:
        eps_history = self.generateEpisode(test_mode)

        replay_buffer.add.remote(eps_history, shared_storage)
        shared_storage.logEpsReward.remote(eps_history.reward_history[-1])
        gc.collect()
      else:
        eps_history = self.generateEpisode(test_mode)
        past_100_rewards = ray.get(shared_storage.getInfo.remote('past_100_rewards'))
        past_100_rewards.append(eps_history.reward_history[-1])

        shared_storage.setInfo.remote(
          {
            'eps_len' : len(eps_history.action_history),
            'total_reward' : sum(eps_history.reward_history),
            'past_100_rewards': past_100_rewards,
            'mean_value' : np.mean([value for value in eps_history.value_history]),
            'eps_obs' : [eps_history.obs_history, eps_history.pred_obs_history],
            'eps_values' : [np.round(value, 2) for value in eps_history.value_history],
            'eps_sampled_actions' : eps_history.sampled_action_history,
            'eps_q_maps' : eps_history.q_map_history
          }
        )
        gc.collect()

      if not test_mode and self.config.data_gen_delay:
        time.sleep(self.config.gen_delay)
      if not test_mode and self.config.train_data_ratio:
        while(
            ray.get(shared_storage.getInfo.remote('training_step'))
            / max(1, ray.get(shared_storage.getInfo.remote('num_steps')))
            < self.config.train_data_ratio):
          time.sleep(0.5)

  def generateEpisode(self, test_mode):
    eps_history = EpisodeHistory()

    obs = self.env.reset()
    obs_rb = (obs[0], self.agent.preprocessDepth(obs[1]), self.agent.preprocessDepth(obs[2]))
    if self.config.agent == 'adn':
      with torch.no_grad():
        value, reward = self.agent.state_value_model(
          torch.Tensor(obs_rb[2]).to(self.device),
          torch.Tensor(obs_rb[1]).to(self.device)
        )
      eps_history.value_history.append(value.item())
    else:
      eps_history.value_history.append(0)
    eps_history.action_history.append([0,0,0,0])
    eps_history.obs_history.append(obs_rb)
    eps_history.reward_history.append(0.0)

    done = False
    while (not done):
      state = int(obs[0])
      if self.config.agent == 'fc_dqn' or self.config.agent == 'rot_fc_dqn':
        q_map, pixel_action, value = self.agent.selectAction(obs)
        sampled_actions = pixel_action.view(1,-1)
        pred_obs = None
      elif self.config.agent == 'shooting':
        q_map, pixel_action, pred_obs = self.agent.selectAction(obs)
        sampled_actions = pixel_action.view(1,-1)
        value = 0
      else:
        q_map, q_maps, sampled_actions, pixel_action, pred_obs, value = self.agent.selectAction(obs)
      pixel_action = pixel_action.tolist()
      action = utils.getWorkspaceAction(pixel_action, constants.WORKSPACE, constants.OBS_RESOLUTION, self.agent.rotations)

      obs, reward, done = self.env.step(action.cpu().numpy(), auto_reset=False)

      if np.max(obs[2]) > self.config.max_height:
        done = True
        continue

      obs_rb = [obs[0], self.agent.preprocessDepth(obs[1]), self.agent.preprocessDepth(obs[2])]
      eps_history.value_history.append(value)
      eps_history.sampled_action_history.append(sampled_actions)
      eps_history.action_history.append(pixel_action)
      eps_history.obs_history.append(obs_rb)
      if test_mode:
        if pred_obs is not None:
          eps_history.pred_obs_history.append(
            data_utils.convertProbToDepth(pred_obs, self.config.num_depth_classes).squeeze().cpu().numpy()
          )
        else:
          eps_history.pred_obs_history.append(obs_rb[2].squeeze())
        eps_history.q_map_history.append(q_map.cpu().numpy().squeeze())
      eps_history.reward_history.append(reward)

    eps_history.q_map_history.append(np.zeros((self.config.obs_size, self.config.obs_size)))
    eps_history.sampled_action_history.append(None)

    return eps_history

  def continuousSimDataGen(self, shared_storage, replay_buffer):
    while (
      ray.get(shared_storage.getInfo.remote('training_step')) < self.config.training_steps and \
      not ray.get(shared_storage.getInfo.remote('terminate'))
    ):
      # NOTE: This might be a inificent but we can't check the training step really due to async
      self.agent.setWeights(ray.get(shared_storage.getInfo.remote('weights')))

      eps_history = self.generateSimEpisode()
      replay_buffer.add.remote(eps_history)
      gc.collect()

  def generateSimEpisode(self):
    eps_history = EpisodeHistory()

    obs = self.env.reset()
    obs_rb = (obs[0], self.agent.preprocessDepth(obs[1]), self.agent.preprocessDepth(obs[2]))
    if self.config.agent == 'adn':
      with torch.no_grad():
        value, reward = self.agent.state_value_model(
          torch.Tensor(obs_rb[2]).to(self.device),
          torch.Tensor(obs_rb[1]).to(self.device)
        )
      eps_history.value_history.append(value.item())
    else:
      eps_history.value_history.append(0)
    eps_history.action_history.append([0,0,0,0])
    eps_history.obs_history.append(obs_rb)
    eps_history.reward_history.append(0.0)

    done = False
    while (not done):
      state, hand_obs, obs = int(obs[0]), obs[1], obs[2]
      q_map, pixel_action, value = self.agent.selectAction((state, hand_obs, obs))
      pixel_action = pixel_action.tolist()

      rotations = self.agent.rotations.to(self.device)
      action = torch.tensor([pixel_action[0],
                             pixel_action[2],
                             pixel_action[1],
                             rotations[pixel_action[3].long()]]).view(1, 4)
      deictic_obs = utils.getDeicticActions(obs, action[:,1:])
      deictic_obs_, obs_, = self.forward_model(deictic_obs,
                                              hand_obs,
                                              obs,
                                              action[:,1:])

      deictic_obs_ = deictic_obs_[torch.arange(self.config.batch_size), state]
      obs_ = obs_[torch.arange(obs_.size(0)), state]

      hand_obs_ = utils.getHandObs(deictic_obs).cpu()
      state_ = self.agent.getHandStates(deictic_obs.cpu(), deictic_obs_.cpu())

      if torch.sum((~state_).long()) > 0:
        num_hand_obs_empty = torch.sum((~state_).long()).item()
        empty_hand_obs = torch.zeros(num_hand_obs_empty, 1, self.agent.hand_obs_size, self.agent.hand_obs_size)
        empty_idx = torch.where(state_.int() == 0)
        hand_obs_[empty_idx] = data_utils.convertDepthToOneHot(empty_hand_obs,
                                                               self.config.num_depth_classes)

      _, reward = self.agent.reward_model(obs_, hand_obs_)
      done = (reward > 0.9)

      obs_rb = [state_, hand_obs_, obs_]
      eps_history.value_history.append(0)
      eps_history.action_history.append(pixel_action)
      eps_history.obs_history.append(obs_rb)
      eps_history.reward_history.append(reward)

    return eps_history

@ray.remote
class ExpertDataGenerator(object):
  def __init__(self, initial_checkpoint, config, seed):
    self.seed = seed
    self.config = config
    self.device = torch.device('cpu')

    env_type, env_config, planner_config = data_utils.getEnvConfig(self.config.expert_env,
                                                                   self.config.use_rot)
    self.env = env_factory.createEnvs(0, 'pybullet', env_type, env_config, planner_config=planner_config)

    self.preprocessDepth = functools.partial(data_utils.preprocessDepth,
                                             min=0.,
                                             max=self.config.max_height,
                                             num_classes=self.config.num_depth_classes,
                                             round=2,
                                             noise=False)

    npr.seed(self.seed)
    torch.manual_seed(self.seed)

  def continuousDataGen(self, shared_storage, replay_buffer):
    while (
      ray.get(shared_storage.getInfo.remote('training_step')) < self.config.training_steps and \
      not ray.get(shared_storage.getInfo.remote('terminate'))
    ):
      if 'deconstruct' in self.config.expert_env:
        valid_eps, eps_history = self.generateEpisodeWithDeconstruct()
      else:
        valid_eps, eps_history = self.generateEpisode()

      if len(eps_history.obs_history) == 1 or not valid_eps:
        continue
      else:
        replay_buffer.add.remote(eps_history, shared_storage)

    ray.actor.exit_actor()

  def generateEpisodeWithDeconstruct(self):
    eps_history = EpisodeHistory(expert_traj=True)
    actions = list()
    obs = self.env.reset()

    # Deconstruct structure while saving the actions reversing the action primative
    done = False
    while not done:
      action = self.env.getNextAction()

      primative = np.abs(action[0] - 1)
      actions.append([primative, action[1], action[2], action[3]])

      obs, reward, done = self.env.step(action, auto_reset=False)

    obs_rb = [obs[0], self.preprocessDepth(obs[1]), self.preprocessDepth(obs[2])]
    valid_eps = self.env.isSimValid()

    eps_history.value_history.append(0.0)
    eps_history.action_history.append([0,0,0,0])
    eps_history.obs_history.append(obs_rb)
    eps_history.reward_history.append(0)

    #import matplotlib.pyplot as plt
    #fig, ax = plt.subplots(nrows=1, ncols=2)
    #ax[0].imshow(obs_rb[2].squeeze(), cmap='gray')
    #ax[1].imshow(obs_rb[1].squeeze(), cmap='gray')
    #plt.show()

    for i, action in enumerate(actions[::-1]):
      rotations = torch.from_numpy(np.linspace(0, np.pi, self.config.num_rots, endpoint=False))
      rot_idx = np.abs(rotations - action[3]).argmin()
      action[-1] = rotations[rot_idx]

      pixel_action = utils.getPixelAction(action, self.config.workspace, self.config.obs_resolution, self.config.obs_size).tolist()
      pixel_action = [pixel_action[0], pixel_action[2], pixel_action[1], rot_idx]

      #deictic_obs = utils.getDeicticActions(
      #  torch.from_numpy(obs_rb[2]).view(1,1,128,128).float(),
      #  torch.tensor([pixel_action[2], pixel_action[1], rotations[rot_idx]]).view(1,3).float()
      #)
      #plt.imshow(deictic_obs.squeeze(), cmap='gray'); plt.show()

      obs, reward, done = self.env.step(action, auto_reset=False)
      obs_rb = [obs[0], self.preprocessDepth(obs[1]), self.preprocessDepth(obs[2])]

      #fig, ax = plt.subplots(nrows=1, ncols=2)
      #ax[0].imshow(obs_rb[2].squeeze(), cmap='gray')
      #ax[1].imshow(obs_rb[1].squeeze(), cmap='gray')
      #plt.show()

      if not self.env.isSimValid():
        break

      if self.env.didBlockFall():
        reward = 0
      else:
        reward = 1 if i == len(actions) - 1 else 0

      eps_history.value_history.append(0.0)
      eps_history.action_history.append(pixel_action)
      eps_history.obs_history.append(obs_rb)
      eps_history.reward_history.append(reward)

    return valid_eps, eps_history

  def generateEpisode(self):
    eps_history = EpisodeHistory(expert_traj=True)

    obs = self.env.reset()
    obs_rb = [obs[0], self.preprocessDepth(obs[1]), self.preprocessDepth(obs[2])]

    eps_history.value_history.append(0.0)
    eps_history.action_history.append([0,0,0,0])
    eps_history.obs_history.append(obs_rb)
    eps_history.reward_history.append(0)

    done = False

    #import matplotlib.pyplot as plt
    #fig, ax = plt.subplots(nrows=1, ncols=2)
    #ax[0].imshow(obs_rb[2].squeeze(), cmap='gray')
    #ax[1].imshow(obs_rb[1].squeeze(), cmap='gray')
    #plt.show()

    while not done:
      action = self.env.getNextAction()
      rotations = torch.from_numpy(np.linspace(0, np.pi, self.config.num_rots, endpoint=False))
      rot_idx = np.abs(rotations - action[3]).argmin()
      action[-1] = rotations[rot_idx]

      obs, reward, done = self.env.step(action, auto_reset=False)

      pixel_action = utils.getPixelAction(action, self.config.workspace, self.config.obs_resolution, self.config.obs_size).tolist()
      pixel_action = [pixel_action[0], pixel_action[2], pixel_action[1], rot_idx]

      #deictic_obs = utils.getDeicticActions(
      #  torch.from_numpy(obs_rb[2]).view(1,1,128,128).float(),
      #  torch.tensor([pixel_action[2], pixel_action[1], rotations[rot_idx]]).view(1,3).float()
      #)
      #plt.imshow(deictic_obs.squeeze(), cmap='gray'); plt.show()

      obs_rb = [obs[0], self.preprocessDepth(obs[1]), self.preprocessDepth(obs[2])]

      #import matplotlib.pyplot as plt
      #fig, ax = plt.subplots(nrows=1, ncols=2)
      #ax[0].imshow(obs_rb[2].squeeze(), cmap='gray')
      #ax[1].imshow(obs_rb[1].squeeze(), cmap='gray')
      #plt.show()

      eps_history.value_history.append(0.0)
      eps_history.action_history.append(pixel_action)
      eps_history.obs_history.append(obs_rb)
      eps_history.reward_history.append(reward)

    return True, eps_history
