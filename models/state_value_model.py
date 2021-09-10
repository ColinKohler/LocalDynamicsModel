import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_utils import Flatten, makeLayer, BasicBlock
from data import constants
from data import data_utils

class StateValueModel(nn.Module):
  def __init__(self, in_channels, device):
    super(StateValueModel, self).__init__()
    self.device = device

    self.in_channels = in_channels

    self.obs_feat = nn.Sequential(
      makeLayer(BasicBlock, self.in_channels, 32, 1, stride=2, bnorm=False),
      makeLayer(BasicBlock, 32, 64, 1, stride=2, bnorm=False),
      makeLayer(BasicBlock, 64, 128, 1, stride=2, bnorm=False),
      makeLayer(BasicBlock, 128, 256, 1, stride=2, bnorm=False)
    )

    self.hand_feat = nn.Sequential(
      makeLayer(BasicBlock, self.in_channels, 32, 1, stride=2, bnorm=False),
      makeLayer(BasicBlock, 32, 64, 1, stride=2, bnorm=False),
      makeLayer(BasicBlock, 64, 128, 1, stride=2, bnorm=False)
    )

    self.value_head = nn.Sequential(
      nn.Conv2d(256+128, 256, 3, stride=2, padding=1),
      nn.LeakyReLU(0.01, inplace=True),
      Flatten(),
      nn.Linear(256*4*4, 2048),
      nn.LeakyReLU(0.01, inplace=True),
      nn.Linear(2048, 1028),
      nn.LeakyReLU(0.01, inplace=True),
      nn.Linear(1028, 256),
      nn.LeakyReLU(0.01, inplace=True),
      nn.Linear(256, 1),
    )

    self.reward_head = nn.Sequential(
      nn.Conv2d(256+128, 256, 3, stride=2, padding=1),
      nn.LeakyReLU(0.01, inplace=True),
      Flatten(),
      nn.Linear(256*4*4, 2048),
      nn.LeakyReLU(0.01, inplace=True),
      nn.Linear(2048, 1028),
      nn.LeakyReLU(0.01, inplace=True),
      nn.Linear(1028, 256),
      nn.LeakyReLU(0.01, inplace=True),
      nn.Linear(256, 1),
    )

    for m in self.modules():
      if isinstance(m, (nn.Conv2d)):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', a=0.01, nonlinearity='leaky_relu')
      elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

  def forward(self, obs, hand_obs):
    batch_size = obs.size(0)

    if obs.size(1) > self.in_channels:
      obs = data_utils.convertProbToDepth(obs, obs.size(1))
      hand_obs = data_utils.convertProbToDepth(hand_obs, hand_obs.size(1))

    pad = int((constants.DEICTIC_OBS_SIZE - hand_obs.size(-1)) / 2)
    hand_obs = F.pad(hand_obs, [pad] * 4)

    obs_feat = self.obs_feat(obs)
    hand_feat = self.hand_feat(hand_obs)
    state_feat = torch.cat((obs_feat, hand_feat), dim=1)

    value = self.value_head(state_feat)
    reward = self.reward_head(state_feat)

    return value, reward

  def loadModel(self, model_state_dict):
    self_state = self.state_dict()
    for name, param in model_state_dict.items():
      self_state[name].copy_(param)
