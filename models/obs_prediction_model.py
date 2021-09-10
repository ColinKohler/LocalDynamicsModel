import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.torch_utils import Flatten, makeLayer, BottleneckBlock, UpsamplingBlock, CatConv
from data import constants

class ObsPredictionModel(nn.Module):
  def __init__(self, device, out_kernels):
    super(ObsPredictionModel, self).__init__()
    self.device = device

    self.pick_model = ActionPrimativeModel(device, out_kernels)
    self.place_model = ActionPrimativeModel(device, out_kernels)

  def forward(self, deictic_obs, hand_obs, obs, action):
    batch_size = obs.size(0)

    pick_deictic_obs_, pick_obs_ = self.pick_model(deictic_obs, hand_obs, obs, action)
    place_deictic_obs_, place_obs_ = self.place_model(deictic_obs, hand_obs, obs, action)

    deictic_obs_ = torch.cat((pick_deictic_obs_.unsqueeze(1), place_deictic_obs_.unsqueeze(1)), dim=1)
    obs_ = torch.cat((pick_obs_.unsqueeze(1), place_obs_.unsqueeze(1)), dim=1)

    return deictic_obs_, obs_

  def loadModel(self, model_state_dict):
    self_state = self.state_dict()
    for name, param in model_state_dict.items():
      self_state[name].copy_(param)

class ActionPrimativeModel(nn.Module):
  def __init__(self, device, out_kernels):
    super(ActionPrimativeModel, self).__init__()
    self.device = device

    self.in_kernels = 2 * out_kernels
    self.out_kernels = out_kernels

    input_kernels = 32
    # TODO: This should just be a nn.Sequential
    self.conv_1 = nn.Conv2d(self.in_kernels, input_kernels, 3, stride=1, padding=1, bias=False)
    self.bn1 = nn.BatchNorm2d(input_kernels)
    self.relu = nn.LeakyReLU(0.01, inplace=True)

    self.layer_1 = makeLayer(BottleneckBlock, 32, 32, 1, stride=2)
    self.layer_2 = makeLayer(BottleneckBlock, 64, 64, 2, stride=2)
    self.layer_3 = makeLayer(BottleneckBlock, 128, 128, 3, stride=2)
    self.layer_4 = makeLayer(BottleneckBlock, 256, 256, 1, stride=2)

    self.forward_layer = nn.Sequential(
      nn.Conv2d(512, 512, 3, stride=1, padding=1, bias=True),
      nn.BatchNorm2d(512),
      nn.LeakyReLU(0.01, inplace=True),
    )

    self.up_proj_4 = UpsamplingBlock(512, 512)
    self.up_proj_3 = UpsamplingBlock(256, 256)
    self.up_proj_2 = UpsamplingBlock(128, 128)
    self.up_proj_1 = UpsamplingBlock(64, 64)

    self.cat_conv_4 = CatConv(512, 256, 256)
    self.cat_conv_3 = CatConv(256, 128, 128)
    self.cat_conv_2 = CatConv(128, 64, 64)
    self.cat_conv_1 = CatConv(64, 32, 32)

    self.out = nn.Conv2d(32, self.out_kernels, kernel_size=3, stride=1, padding=1, bias=True)
    self.softmax = nn.Softmax(dim=1)

    for m in self.modules():
      if isinstance(m, (nn.Conv2d)):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', a=0.01, nonlinearity='leaky_relu')
      elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

  def forward(self, deictic_obs, hand_obs, obs, action):
    batch_size = obs.size(0)
    hand_obs = self.padHandObs(hand_obs)

    inp = torch.cat((deictic_obs, hand_obs), dim=1)
    inp = self.conv_1(inp)
    inp = self.bn1(inp)
    inp = self.relu(inp)

    feat_down_1 = self.layer_1(inp)
    feat_down_2 = self.layer_2(feat_down_1)
    feat_down_3 = self.layer_3(feat_down_2)
    feat_down_4 = self.layer_4(feat_down_3)

    forward_feat = self.forward_layer(feat_down_4)

    feat_up_4 = self.cat_conv_4(self.up_proj_4(forward_feat), feat_down_3)
    feat_up_3 = self.cat_conv_3(self.up_proj_3(feat_up_4), feat_down_2)
    feat_up_2 = self.cat_conv_2(self.up_proj_2(feat_up_3), feat_down_1)
    feat_up_1 = self.cat_conv_1(self.up_proj_1(feat_up_2), inp)

    deictic_obs_ = self.out(feat_up_1)
    deictic_obs_ = self.softmax(deictic_obs_)
    obs_ = self.replaceObs(obs, deictic_obs_.detach(), action)

    return deictic_obs_, obs_

  def padHandObs(self, hand_obs):
    batch_size = hand_obs.size(0)

    pad_size = (batch_size, self.out_kernels, constants.DEICTIC_OBS_SIZE, constants.DEICTIC_OBS_SIZE)
    hand_obs_pad = torch.zeros(pad_size).float().to(self.device)
    hand_obs_pad[:,0] = 1.0

    c = round(constants.DEICTIC_OBS_SIZE / 2)
    s = round(hand_obs.size(-1) / 2)
    hand_obs_pad[:, :, c-s:c+s, c-s:c+s] = hand_obs

    return hand_obs_pad

  def replaceObs(self, obs, deictic_obs, actions):
    R = torch.zeros(actions.size(0), 2, 3)
    R[:,0,0] = torch.cos(actions[:,2])
    R[:,0,1] = -torch.sin(actions[:,2])
    R[:,1,0] = torch.sin(actions[:,2])
    R[:,1,1] = torch.cos(actions[:,2])

    grid_shape = (actions.size(0), 1, constants.DEICTIC_OBS_SIZE, constants.DEICTIC_OBS_SIZE)
    grid = F.affine_grid(R, grid_shape, align_corners=True).to(self.device)
    deictic_obs = F.grid_sample(deictic_obs, grid, padding_mode='zeros', align_corners=False, mode='bilinear')

    c = actions[:, :2]
    padding = [int(constants.DEICTIC_OBS_SIZE / 2)] * 4
    padded_obs = F.pad(obs, padding, 'constant', 0.0)
    c = c + int(constants.DEICTIC_OBS_SIZE / 2)

    b = padded_obs.size(0)
    h = padded_obs.size(-1)
    s = int(constants.DEICTIC_OBS_SIZE / 2)
    x = torch.clamp(c[:,0].view(b,1) + torch.arange(-s, s).repeat(b, 1).to(self.device), 0, h-1).long()
    y = torch.clamp(c[:,1].view(b,1) + torch.arange(-s, s).repeat(b, 1).to(self.device), 0, h-1).long()

    ind = torch.transpose(((x * h).repeat(1,s*2).view(b,s*2,s*2) + y.view(b,s*2,1)), 1, 2)

    padded_obs = padded_obs.view(b*self.out_kernels,-1)
    padded_obs.scatter_(1, ind.reshape(b,-1).repeat(1, self.out_kernels).view(b*self.out_kernels,-1), deictic_obs.view(b*self.out_kernels, -1))
    padded_obs = padded_obs.view(b, self.out_kernels, h, h)

    start = int(constants.DEICTIC_OBS_SIZE / 2)
    end = obs.size(3) + int(constants.DEICTIC_OBS_SIZE / 2)
    new_obs = padded_obs[:, :, start:end, start:end]

    return new_obs

  def loadModel(self, model_state_dict):
    self_state = self.state_dict()
    for name, param in model_state_dict.items():
      self_state[name].copy_(param)
