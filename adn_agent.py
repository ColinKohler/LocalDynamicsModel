import sys
sys.path.append('..')

import os
import copy
import numpy as np
import numpy.random as npr
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from agents.base_agent import BaseAgent
from models.obs_prediction_model import ObsPredictionModel
from models.state_value_model import StateValueModel
from models.q_value_model import QValueModel
from data import data_utils
from data.episode_history import Node
import utils

import torch_utils.utils as torch_utils
import torch_utils.losses as torch_losses

class ADNAgent(BaseAgent):
  def __init__(self, config, device, training=False, depth=None):
    super(ADNAgent, self).__init__(config, device, training=training)

    self.branch_factors = [4, 2, 1]
    self.depth = depth if depth else self.config.depth

    self.forward_model = ObsPredictionModel(self.device, self.config.num_depth_classes).to(self.device)
    self.state_value_model = StateValueModel(1, self.device).to(self.device)
    self.q_value_model = QValueModel(1, 1, self.device).to(self.device)
    self.action_sample_pen_size = self.config.init_action_sample_pen_size

    if self.training:
      self.forward_optimizer = torch.optim.Adam(self.forward_model.parameters(),
                                                lr=self.config.forward_lr_init,
                                                weight_decay=self.config.forward_weight_decay,
                                                betas=(0.9, 0.999))
      self.forward_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.forward_optimizer,
                                                                      self.config.lr_decay)

      self.state_value_optimizer = torch.optim.Adam(self.state_value_model.parameters(),
                                                    lr=self.config.state_value_lr_init,
                                                    weight_decay=self.config.state_value_weight_decay,
                                                    betas=(0.9, 0.999))
      self.state_value_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.state_value_optimizer,
                                                                          self.config.lr_decay)

      self.q_value_optimizer = torch.optim.Adam(self.q_value_model.parameters(),
                                                lr=self.config.q_value_lr_init,
                                                weight_decay=self.config.q_value_weight_decay,
                                                betas=(0.9, 0.999))
      self.q_value_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.q_value_optimizer,
                                                                      self.config.lr_decay)
      self.focal_loss = torch_losses.FocalLoss(self.device,
                                               alpha=torch.ones(self.config.num_depth_classes),
                                               gamma=0.0,
                                               smooth=1e-5,
                                               size_average=True)

      self.forward_model.train()
      self.state_value_model.train()
      self.q_value_model.train()
    else:
      self.forward_model.eval()
      self.state_value_model.eval()
      self.q_value_model.eval()

  def selectAction(self, obs, normalize_obs=True, return_all_states=False):
    state, hand_obs, obs = obs

    obs = torch.Tensor(obs.astype(np.float32)).view(1, 1, self.config.obs_size, self.config.obs_size)
    hand_obs = torch.Tensor(hand_obs.astype(np.float32)).view(1, 1, self.config.hand_obs_size, self.config.hand_obs_size)
    if normalize_obs:
      obs = data_utils.convertDepthToOneHot(self.preprocessDepth(obs),
                                            self.config.num_depth_classes)
      hand_obs = data_utils.convertDepthToOneHot(self.preprocessDepth(hand_obs),
                                                 self.config.num_depth_classes)
    else:
      obs = data_utils.convertDepthToOneHot(obs, self.config.num_depth_classes)
      hand_obs = data_utils.convertDepthToOneHot(hand_obs, self.config.num_depth_classes)

    q_map, sampled_actions, action, pred_obs, value = self.getAgentAction((state, hand_obs, obs),
                                                                          return_all_states=return_all_states)

    action = torch.Tensor([state, action[0], action[1], action[2]])
    return q_map[action[3].long(),:], q_map, sampled_actions, action, pred_obs, value

  def getAgentAction(self, obs, return_all_states=False):
    with torch.no_grad():
      value, reward = self.state_value_model(obs[2].to(self.device), obs[1].to(self.device))

    rot_obs = utils.rotateObs(obs[2].repeat(self.config.num_rots, 1, 1, 1), self.rotations)
    hand_obs = obs[1].repeat(self.config.num_rots, 1, 1, 1)
    with torch.no_grad():
      rot_q_map = self.q_value_model(rot_obs.to(self.device), hand_obs.to(self.device))

    rot_q_map = rot_q_map[:, int(obs[0])]
    q_map = utils.rotateObs(rot_q_map, -self.rotations)
    #q_map = self.getQMap(obs)
    root = Node(None, 0, obs, value.item(), reward.item(), q_map.cpu().numpy())

    self.expandNode(root)
    child_nodes = list(root.children.values())
    children_values = [n.value for n in child_nodes]
    best_node = child_nodes[np.argmax(children_values)]

    if return_all_states:
      return q_map, [n.parent[1] for n in child_nodes], best_node.parent[1], [n.obs[2] for n in child_nodes], [n.value for n in child_nodes]
    else:
      return q_map, root.sampled_actions, best_node.parent[1], best_node.obs[2], best_node.value

  def expandNode(self, node):
    # Get deictic actions at each sampled action position
    pixel_actions = self.sampleActionsFromQMap(node.q_map)
    node.sampled_actions = pixel_actions

    action_hand_obs = node.obs[1].repeat(self.config.num_sampled_actions, 1, 1, 1)
    action_obs = node.obs[2].repeat(self.config.num_sampled_actions, 1, 1, 1)
    pixel_actions_w_rot = torch.stack([pixel_actions[:,0],
                                       pixel_actions[:,1],
                                       self.rotations[pixel_actions[:,2].long()]])
    pixel_actions_w_rot = pixel_actions_w_rot.permute(1,0)

    deictic_pixel_actions = torch.stack([pixel_actions[:,1],
                                         pixel_actions[:,0],
                                         self.rotations[pixel_actions[:,2].long()]])
    deictic_pixel_actions = deictic_pixel_actions.permute(1,0)

    deictic_obs = utils.getDeicticActions(action_obs, deictic_pixel_actions)

    # Use forward model to get the state for all actions being considered
    with torch.no_grad():
      deictic_obs_, action_obs_ = self.forward_model(deictic_obs.to(self.device),
                                                     action_hand_obs.to(self.device),
                                                     action_obs.to(self.device),
                                                     pixel_actions_w_rot.to(self.device))

    deictic_obs_ = deictic_obs_[:,int(node.obs[0])]
    action_obs_ = action_obs_[:,int(node.obs[0])]

    # Get the new hand obs by checking the hand state at the previous timestep
    hand_obs_ = utils.getHandObs(deictic_obs).cpu()
    state_ = self.getHandStates(deictic_obs.cpu(), deictic_obs_.cpu())

    # Depending on the state of the hand (holding, not_holding) we set the hand obs to
    # ether a blank obs or the deictic obs that was executed
    if torch.sum((~state_).long()) > 0:
      num_hand_obs_empty = torch.sum((~state_).long()).item()
      empty_hand_obs = torch.zeros(num_hand_obs_empty, 1, self.hand_obs_size, self.hand_obs_size)
      empty_idx = torch.where(state_.int() == 0)
      hand_obs_[empty_idx] = data_utils.convertDepthToOneHot(empty_hand_obs,
                                                             self.config.num_depth_classes)

    # Use value model to get value predictions for the new states
    with torch.no_grad():
      action_value, action_reward = self.state_value_model(action_obs_.to(self.device),
                                                           hand_obs_.to(self.device))
      action_q_map = self.q_value_model(action_obs_.to(self.device),
                                        hand_obs_.to(self.device))

    action_value = action_value.squeeze()
    action_reward = action_reward.squeeze()

    #for v, o, h in zip(action_value, action_obs_, hand_obs_):
     # utils.plotObs(o,h,v)

    # Add nodes to the search tree
    node.expand(pixel_actions,
                state_,
                hand_obs_,
                action_obs_,
                action_value,
                action_reward,
                action_q_map)

  def sampleActionsFromQMap(self, q_map):
    pen_size = self.action_sample_pen_size[1]
    rot_pen = self.action_sample_pen_size[0]
    pen = utils.generate2dGaussian(pen_size+1, 0., 1.)

    q_map = torch.from_numpy(q_map).squeeze()
    q_map_pad = F.pad(q_map, (int(pen_size/2), int(pen_size/2), int(pen_size/2), int(pen_size/2)), value=-1)

    actions = list()
    while len(actions) < self.config.num_sampled_actions:
      pad_a = torch_utils.argmax3d(q_map_pad.view(1, self.config.num_rots, q_map_pad.size(-1), q_map_pad.size(-1))).long()

      a = [pad_a[0,1] - int(pen_size/2), pad_a[0,2] - int(pen_size/2), pad_a[0,0]]
      if a not in actions:
        actions.append(a)

      # TODO: This should be vectorized
      rots = torch.arange(torch.clamp(pad_a[0,0].long() - rot_pen, 0, self.config.num_rots - 1),
                          torch.clamp(pad_a[0,0].long() + rot_pen, 0, self.config.num_rots - 1) + 1)
      for r in rots:
        q_map_pad[r,
                  pad_a[0,1] - int(pen_size/2):pad_a[0,1] + int(pen_size/2) + 1,
                  pad_a[0,2] - int(pen_size/2):pad_a[0,2] + int(pen_size/2) + 1] -= pen

    pixel_actions = torch.tensor(actions)

    return pixel_actions

  def decayActionSamplePen(self):
    rot_pen = self.action_sample_pen_size[0]
    pos_pen = self.action_sample_pen_size[1]

    self.action_sample_pen_size = [max(rot_pen - 1, self.config.end_action_sample_pen_size[0]),
                                   max(pos_pen - 2, self.config.end_action_sample_pen_size[1])]

  def getBranchFactor(self, depth):
    if self.depth == 1:
      return self.config.num_sampled_actions
    elif depth >= len(self.branch_factors):
      return self.branch_factors[-1]
    else:
      return self.branch_factors[depth]

  def updateWeights(self, batch, class_weight):
    # Check that training mode was enabled at init
    if not self.training:
      return None

    # Process batch and load onto device
    state_batch, hand_obs_batch, obs_batch, action_batch, target_state_value, target_q_value, target_reward, weight_batch, = batch

    target_state_value_scalar = np.array(target_state_value, dtype=np.float32)
    td_error = np.zeros_like(target_state_value_scalar)

    state_batch = state_batch.to(self.device)
    hand_obs_batch = hand_obs_batch.to(self.device)
    obs_batch = obs_batch.to(self.device)
    action_batch = action_batch.to(self.device)
    target_state_value = target_state_value.to(self.device)
    target_q_value = target_q_value.to(self.device)
    target_reward = target_reward.to(self.device)
    weight_batch = weight_batch.to(self.device)

    # Inital value prediction for t=0
    pred_state_value, pred_reward = self.state_value_model(obs_batch[:,0], hand_obs_batch[:,0])
    rot_obs_batch = utils.rotateObs(obs_batch[:,0], self.rotations[action_batch[:,1,3].long()])
    pred_rot_q_map = self.q_value_model(rot_obs_batch, hand_obs_batch[:,0])
    pred_rot_q_map = pred_rot_q_map[torch.arange(self.config.batch_size),
                                    action_batch[:,1,0].long()]
    pred_q_map = utils.rotateObs(pred_rot_q_map, -self.rotations[action_batch[:,1,3].long()])

    #pred_q_map = self.getQMap((state_batch[:,0], hand_obs_batch[:,0], obs_batch[:,0]), grad=True)
    pred_q_value = pred_q_map[torch.arange(self.config.batch_size),
                              0,
                              action_batch[:,1,1].long(),
                              action_batch[:,1,2].long()]

    predictions = [(None, pred_q_value, pred_state_value, pred_reward)]
    deictic_labels = [None]

    # Value and forward predictions for t>0
    obs = obs_batch[:,0]
    rotations = self.rotations.to(self.device)
    for t in range(1, action_batch.size(1)):
      action = torch.stack([action_batch[:,t,0],
                            action_batch[:,t,2],
                            action_batch[:,t,1],
                            rotations[action_batch[:,t,3].long()]]).permute(1,0)

      deictic_obs = utils.getDeicticActions(obs, action[:,1:])
      deictic_obs_, obs, = self.forward_model(deictic_obs,
                                              hand_obs_batch[:,t-1],
                                              obs,
                                              action[:,1:])

      deictic_obs_ = deictic_obs_[torch.arange(self.config.batch_size), state_batch[:,t-1]]
      obs = obs[torch.arange(self.config.batch_size), state_batch[:,t-1]]

      pred_state_value, pred_reward = self.state_value_model(obs, hand_obs_batch[:,t])
      #pred_q_map = self.getQMap((state_batch, hand_obs_batch, obs_batch))
      pred_q_map = None

      predictions.append((deictic_obs_, pred_q_map, pred_state_value, pred_reward))

      deictic_target = utils.getDeicticActions(obs_batch[:,t], action[:,1:])
      deictic_labels.append(data_utils.convertProbToLabel(deictic_target, self.config.num_depth_classes))

    # Compute losses
    q_value_loss, state_value_loss, reward_loss, forward_loss = 0, 0, 0, 0
    for t in range(len(predictions)):
      pred_deictic_obs, pred_q_value, pred_state_value, pred_reward = predictions[t]
      deictic_label = deictic_labels[t].to(self.device) if deictic_labels[t] is not None else None

      q_loss, v_loss, r_loss, f_loss = self.lossFunction(pred_q_value,
                                                         pred_state_value.squeeze(),
                                                         pred_reward.squeeze(),
                                                         pred_deictic_obs,
                                                         target_state_value[:,t],
                                                         target_q_value[:,t+1] if t==0 else t,
                                                         target_reward[:,t],
                                                         deictic_label,
                                                         class_weight.to(self.device))

      q_value_loss += q_loss
      state_value_loss += v_loss
      reward_loss += r_loss
      forward_loss += f_loss

      pred_state_value_scalar = pred_state_value.detach().cpu().numpy().squeeze()
      td_error[:,t] = np.abs(pred_state_value_scalar - target_state_value_scalar[:,t])

    # Compute the weighted losses using the PER weights
    q_value_loss = (q_value_loss * weight_batch).mean()
    state_value_loss = (state_value_loss * weight_batch).mean()
    reward_loss = (reward_loss * weight_batch).mean()
    forward_loss = (forward_loss * weight_batch).mean()

    # Optimize
    self.forward_optimizer.zero_grad()
    forward_loss.backward()
    self.forward_optimizer.step()

    self.state_value_optimizer.zero_grad()
    (state_value_loss + reward_loss).backward()
    self.state_value_optimizer.step()

    self.q_value_optimizer.zero_grad()
    q_value_loss.backward()

    for param in self.q_value_model.parameters():
      param.grad.data.clamp_(-1, 1)
    self.q_value_optimizer.step()

    pred_deictic_obs, pred_q_value, pred_state_value, pred_reward = predictions[1]
    pred_obs = [[pred_deictic_obs[0].detach().cpu(), deictic_labels[1][0].detach().cpu()],
                [pred_state_value[0].detach().cpu().item(), target_state_value_scalar[0,1]]]

    return (td_error,
            q_value_loss.item(),
            state_value_loss.item(),
            reward_loss.item(),
            forward_loss.item(),
            pred_obs)

  def lossFunction(self, q_values, state_value, reward, deictic_obs, target_state_value, target_q_value, target_reward, target_deictic_obs, class_weight):
    batch_size = target_state_value.size(0)

    state_value_loss =  F.smooth_l1_loss(state_value, target_state_value, reduction='none')
    reward_loss =  F.smooth_l1_loss(reward, target_reward, reduction='none')
    if q_values is not None:
      #q_values = list()
      #for i in range(batch_size):
      #  actions = target_q_value[i,:,1:]
      #  num_sampled_actions = 1 #self.config.num_sampled_actions + 1
      #  q_values.append(q_map[torch.ones(num_sampled_actions).long() * i, actions[:,2].long(), actions[:,0].long(), actions[:,1].long()])
      #q_values = torch.stack(q_values)
      q_value_loss = F.smooth_l1_loss(q_values, target_q_value, reduction='none')
    else:
      q_value_loss = 0

    if deictic_obs is not None:
      forward_loss = self.focal_loss(deictic_obs, target_deictic_obs, alpha=class_weight)
    else:
      forward_loss = 0

    return q_value_loss, state_value_loss, reward_loss, forward_loss

  def updateLR(self):
    self.forward_scheduler.step()
    self.state_value_scheduler.step()
    self.q_value_scheduler.step()

  def getLR(self):
    return (self.forward_optimizer.param_groups[0]['lr'],
            self.state_value_optimizer.param_groups[0]['lr'],
            self.q_value_optimizer.param_groups[0]['lr'])

  def getWeights(self):
    return (torch_utils.dictToCpu(self.forward_model.state_dict()),
            torch_utils.dictToCpu(self.state_value_model.state_dict()),
            torch_utils.dictToCpu(self.q_value_model.state_dict()))

  def setWeights(self, weights):
    if weights is not None:
      self.forward_model.load_state_dict(weights[0])
      self.state_value_model.load_state_dict(weights[1])
      self.q_value_model.load_state_dict(weights[2])

  def getOptimizerState(self):
    return (torch_utils.dictToCpu(self.forward_optimizer.state_dict()),
            torch_utils.dictToCpu(self.state_value_optimizer.state_dict()),
            torch_utils.dictToCpu(self.q_value_optimizer.state_dict()))

  def setOptimizerState(self, state):
    self.forward_optimizer.load_state_dict(copy.deepcopy(state[0]))
    self.state_value_optimizer.load_state_dict(copy.deepcopy(state[1]))
    self.q_value_optimizer.load_state_dict(copy.deepcopy(state[2]))
