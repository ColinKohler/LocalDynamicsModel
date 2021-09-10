import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_utils import Flatten, makeLayer, BasicBlock, UpsamplingBlock3
from data import constants
from data import data_utils

class QValueModel(nn.Module):
  def __init__(self, in_kernels, out_kernels, device):
    super(QValueModel, self).__init__()
    self.device = device

    self.in_kernels = in_kernels
    self.out_kernels = out_kernels

    self.conv_1 = nn.Sequential(
      nn.Conv2d(self.in_kernels, 16, 3, stride=1, padding=1, bias=False),
      nn.LeakyReLU(0.01, inplace=True)
    )

    self.layer_1 = makeLayer(BasicBlock, 16, 32, 1, stride=2, bnorm=False)
    self.layer_2 = makeLayer(BasicBlock, 32, 64, 1, stride=2, bnorm=False)
    self.layer_3 = makeLayer(BasicBlock, 64, 128, 1, stride=2, bnorm=False)
    self.layer_4 = makeLayer(BasicBlock, 128, 128, 1, stride=2, bnorm=False)

    self.conv_2 = nn.Sequential(
      nn.Conv2d(128, 128, 3, stride=1, padding=1, bias=True),
      nn.LeakyReLU(0.01, inplace=True),
    )

    self.up_proj_4 = UpsamplingBlock3(256, 128, bnorm=False)
    self.up_proj_3 = UpsamplingBlock3(192, 64, bnorm=False)
    self.up_proj_2 = UpsamplingBlock3(96, 32, bnorm=False)
    self.up_proj_1 = UpsamplingBlock3(48, 16, bnorm=False)

    self.hand_feat = nn.Sequential(
      nn.Conv2d(self.in_kernels, 32, 3, stride=2, padding=1, bias=False),
      nn.LeakyReLU(0.01, inplace=True),
      nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
      nn.LeakyReLU(0.01, inplace=True),
      nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
      nn.LeakyReLU(0.01, inplace=True),
      nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False),
      nn.LeakyReLU(0.01, inplace=True),
      Flatten(),
      nn.Linear(256*4*4, 1028),
      nn.LeakyReLU(0.01, inplace=True),
      nn.Linear(1028, 16 * 16 * 3 * 3)
    )
    self.relu = nn.LeakyReLU(0.01, inplace=True)

    self.pick_head = nn.Sequential(
      nn.Conv2d(16, 8, 3, stride=1, padding=1, bias=True),
      nn.LeakyReLU(0.01, inplace=True),
      nn.Conv2d(8, self.out_kernels, 1, stride=1, padding=0, bias=True),
    )
    self.place_head = nn.Sequential(
      nn.Conv2d(16, 8, 3, stride=1, padding=1, bias=True),
      nn.LeakyReLU(0.01, inplace=True),
      nn.Conv2d(8, self.out_kernels, 1, stride=1, padding=0, bias=True),
    )

    for m in self.modules():
      if isinstance(m, (nn.Conv2d)):
        nn.init.xavier_normal_(m.weight)
      elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

  def forward(self, obs, hand_obs):
    batch_size = obs.size(0)

    if obs.size(1) > self.in_kernels:
      obs = data_utils.convertProbToDepth(obs, obs.size(1))
      hand_obs = data_utils.convertProbToDepth(hand_obs, hand_obs.size(1))

    pad = int((constants.DEICTIC_OBS_SIZE - hand_obs.size(-1)) / 2)
    hand_obs = F.pad(hand_obs, [pad] * 4)

    inp = self.conv_1(obs)

    feat_down_1 = self.layer_1(inp)
    feat_down_2 = self.layer_2(feat_down_1)
    feat_down_3 = self.layer_3(feat_down_2)
    feat_down_4 = self.layer_4(feat_down_3)

    feat_down = self.conv_2(feat_down_4)

    feat_up_4 = self.up_proj_4(feat_down, feat_down_3)
    feat_up_3 = self.up_proj_3(feat_up_4, feat_down_2)
    feat_up_2 = self.up_proj_2(feat_up_3, feat_down_1)
    feat_up_1 = self.up_proj_1(feat_up_2, inp)

    hand_kernel = self.hand_feat(hand_obs)
    hand_kernel = hand_kernel.reshape(batch_size * 16, 16, 3, 3)

    place_q_map = feat_up_1.reshape(1, batch_size * feat_up_1.size(1), feat_up_1.size(2), feat_up_1.size(3))
    place_q_map = self.relu(F.conv2d(place_q_map, weight=hand_kernel, groups=batch_size, padding=1))
    place_q_map = place_q_map.reshape(batch_size, 16, feat_up_1.size(2), feat_up_1.size(3))
    place_q_map = self.place_head(place_q_map)

    pick_q_map = self.pick_head(feat_up_1)

    q_map = torch.cat((pick_q_map.unsqueeze(1), place_q_map.unsqueeze(1)), dim=1)

    return q_map

  def loadModel(self, model_state_dict):
    self_state = self.state_dict()
    for name, param in model_state_dict.items():
      self_state[name].copy_(param)
