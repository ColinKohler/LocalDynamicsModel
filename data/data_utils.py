import sys
sys.path.append('..')

import torch
import numpy as np
import numpy.random as npr

import torch_utils.utils as torch_utils

from data import constants
import utils

from data.configs.adn.block_stacking_3 import ADNBlockStacking3Config
from data.configs.adn.house_building_2 import ADNHouseBuilding2Config
from data.configs.adn.house_building_3 import ADNHouseBuilding3Config
from data.configs.adn.bottle_tray import ADNBottleTrayConfig
from data.configs.adn.box_palletizing import ADNBoxPalletizingConfig
from data.configs.adn.bin_packing import ADNBinPackingConfig

from data.configs.shooting.block_stacking_3 import ShootingBlockStacking3Config
from data.configs.shooting.house_building_2 import ShootingHouseBuilding2Config

from data.configs.fc_dqn.block_stacking_3 import FCDQNBlockStacking3Config
from data.configs.fc_dqn.house_building_2 import FCDQNHouseBuilding2Config
from data.configs.fc_dqn.house_building_3 import FCDQNHouseBuilding3Config
from data.configs.fc_dqn.bottle_tray import FCDQNBottleTrayConfig
from data.configs.fc_dqn.box_palletizing import FCDQNBoxPalletizingConfig
from data.configs.fc_dqn.bin_packing import FCDQNBinPackingConfig

from data.configs.rot_fc_dqn.block_stacking_3 import RotFCDQNBlockStacking3Config
from data.configs.rot_fc_dqn.house_building_2 import RotFCDQNHouseBuilding2Config
from data.configs.rot_fc_dqn.house_building_3 import RotFCDQNHouseBuilding3Config
from data.configs.rot_fc_dqn.bottle_tray import RotFCDQNBottleTrayConfig
from data.configs.rot_fc_dqn.box_palletizing import RotFCDQNBoxPalletizingConfig
from data.configs.rot_fc_dqn.bin_packing import RotFCDQNBinPackingConfig

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

task_config_dict = {
  'adn_block_stacking_3' : ADNBlockStacking3Config,
  'adn_house_building_2' : ADNHouseBuilding2Config,
  'adn_house_building_3' : ADNHouseBuilding3Config,
  'adn_bottle_tray' : ADNBottleTrayConfig,
  'adn_box_palletizing' : ADNBoxPalletizingConfig,
  'adn_bin_packing' : ADNBinPackingConfig,
  'shooting_block_stacking_3' : ShootingBlockStacking3Config,
  'shooting_house_building_2' : ShootingHouseBuilding2Config,
  'fc_dqn_block_stacking_3' : FCDQNBlockStacking3Config,
  'fc_dqn_house_building_2' : FCDQNHouseBuilding2Config,
  'fc_dqn_house_building_3' : FCDQNHouseBuilding3Config,
  'fc_dqn_bottle_tray' : FCDQNBottleTrayConfig,
  'fc_dqn_box_palletizing' : FCDQNBoxPalletizingConfig,
  'fc_dqn_bin_packing' : FCDQNBinPackingConfig,
  'rot_fc_dqn_block_stacking_3' : RotFCDQNBlockStacking3Config,
  'rot_fc_dqn_house_building_2' : RotFCDQNHouseBuilding2Config,
  'rot_fc_dqn_house_building_3' : RotFCDQNHouseBuilding3Config,
  'rot_fc_dqn_bottle_tray' : RotFCDQNBottleTrayConfig,
  'rot_fc_dqn_box_palletizing' : RotFCDQNBoxPalletizingConfig,
  'rot_fc_dqn_bin_packing' : RotFCDQNBinPackingConfig,
}

def getTaskConfig(agent, task, num_gpus, results_path=None):
  try:
    config = task_config_dict['{}_{}'.format(agent, task)](num_gpus, results_path=results_path)
  except:
    raise ValueError('Invalid task specified')

  return config

def getPlannerConfig(pick_noise, place_noise, rand_action_prob, random_orientation, planner_type=None):
  config = {
    'pick_noise': pick_noise,
    'place_noise': place_noise,
    'rand_place_prob': rand_action_prob,
    'rand_pick_prob': rand_action_prob,
    'gamma': constants.ENV_GAMMA,
    'random_orientation': random_orientation
  }

  if planner_type:
    config['planner_type'] = planner_type

  return config

def getEnvConfig(env_type, use_rot, use_planner_noise=False, render=False):
  if env_type in constants.ENV_CONFIGS:
    env_config = constants.ENV_CONFIGS[env_type]
    env_type = constants.ENV_TYPES[env_type]
  else:
    raise ValueError('Invalid env type specified')

  env_config['render'] = render
  env_config['random_orientation'] = use_rot

  if use_planner_noise:
    planner_config = getPlannerConfig([0, 0.01], [0, 0.01], 0., use_rot)
  else:
    planner_config = getPlannerConfig([0, 0], [0, 0.], 0, use_rot)

  return env_type, env_config, planner_config

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def preprocessDepth(depth, min, max, num_classes, round=2, noise=False):
  depth = normalizeData(depth, min, max)
  if type(depth) is np.ndarray:
    depth = np.round(depth, round)
    depth = utils.smoothDepth(depth, num_classes)
    if noise:
      depth += npr.rand(*depth.shape)
  elif type(depth) is torch.Tensor:
    depth = torch_utils.roundTensor(depth, round)
    depth = utils.smoothDepthTorch(depth, num_classes)
    if noise:
      depth += torch.rand_like(depth) * 0.01
  else:
    ValueError('Data not numpy array or torch tensor')

  return depth

def normalizeData(data, min, max, eps=1e-8):
  return (data - min) / (max - min + eps)

def unnormalizeData(data, min, max, eps=1e-8):
  return data * (max - min + eps) + min

def convertDepthToOneHot(depth, num_labels):
  if depth.ndim == 2 or depth.ndim == 3:
    num_depth = 1
  elif depth.ndim == 4:
    num_depth = depth.shape[0]
  depth_size = depth.shape[-1]

  inds = torch.from_numpy(convertDepthToLabel(depth, num_labels))
  inds = inds.view(num_depth, 1, depth_size, depth_size).long()

  x = torch.FloatTensor(num_depth, num_labels, depth_size, depth_size)
  x.zero_()
  x.scatter_(1, inds, 1)

  return x

def convertDepthToLabel(depth, num_labels):
  bins = np.linspace(0., 1., num_labels, dtype=np.float32)
  inds = np.digitize(depth, bins, right=True)

  return inds

def convertProbToLabel(prob, num_labels):
  return torch.argmax(prob, dim=1).squeeze()

def convertProbToDepth(prob, num_depth_classes):
  if prob.dim() == 3:
    n = 1
    prob_one_hot = prob.argmax(0).float()
  elif prob.dim() == 4:
    n = prob.size(0)
    prob_one_hot = prob.argmax(1).float()
  depth = prob_one_hot * (1 / (num_depth_classes - 1))

  s = prob.size(-1)
  return depth.view(n, 1, s, s)

def normalizeProb(prob):
  if prob.dim() == 3:
    n = 1
    d = prob.size(0)
    s = prob.size(1)
  elif prob.dim() == 4:
    n = prob.size(0)
    d = prob.size(1)
    s = prob.size(2)

  prob = prob.view(n, d, -1)
  max_prob = torch.sum(prob, dim=1)
  return (prob / max_prob).view(n, d, s, s)
