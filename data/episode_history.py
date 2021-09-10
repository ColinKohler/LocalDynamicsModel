import copy
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import numpy as np
import matplotlib.pyplot as plt

from data import data_utils

class EpisodeHistory(object):
  def __init__(self, expert_traj=False):
    self.expert_traj = expert_traj
    self.obs_history = list()
    self.q_map_history = list()
    self.pred_obs_history = list()
    self.action_history = list()
    self.sampled_action_history = list()
    self.reward_history = list()
    self.value_history = list()
    self.child_visits = list()

    self.priorities = None
    self.eps_priority = None

class Node(object):
  def __init__(self, parent, depth, obs, value, reward, q_map=None):
    self.parent = parent
    self.children = dict()
    self.children_values = dict()
    self.obs = obs
    self.value = value
    self.reward = reward
    self.depth = depth
    self.q_map = q_map
    self.sampled_actions = list()

  def expanded(self):
    return len(self.children) > 0

  def expand(self, actions, state_, hand_obs_, obs_, values_, reward_, q_map):
    for i, action in enumerate(actions):
      action = tuple(action.tolist())
      self.children[action] = Node((self, action),
                                    self.depth+1,
                                    (state_[i], hand_obs_[i], obs_[i]),
                                    values_[i].item(),
                                    reward_[i].item(),
                                    q_map[i].cpu().numpy())
      self.children_values[action] = values_[i].item()

  def isTerminal(self):
    return self.reward >= 1.

  def plot(self, g=None, states=dict(), parent_id=None, node_id=0):
    if g is None:
      g = nx.DiGraph()
    g.add_node(node_id)
    states[node_id] = [
      self.value,
      self.reward,
      data_utils.convertProbToDepth(self.obs[1], 21).cpu().squeeze(),
      data_utils.convertProbToDepth(self.obs[2], 21).cpu().squeeze()
    ]
    if parent_id is not None:
      g.add_edge(parent_id, node_id)

    for i, child in enumerate(self.children.values()):
      child_id = ((node_id + 1) * 10) + i
      child.plot(g=g, states=states, parent_id=node_id, node_id=child_id)

    if parent_id is None:
      pos = graphviz_layout(g, prog='dot')
      nx.draw_networkx(g, pos, width=1, edge_color='r', alpha=0.6)
      fig, ax = plt.gcf(), plt.gca()
      trans, trans2 = ax.transData.transform, fig.transFigure.inverted().transform
      img_size = 0.075

      for n in g.nodes():
        (x, y) = pos[n]
        xx, yy = trans((x, y)) # Figure cords
        xa, ya = trans2((xx, yy)) # Axes cords

        a1 = plt.axes([xa - img_size / 2.0, ya - img_size / 2.0, img_size, img_size])
        a1.set_title('V: {:.2f} | R: {:.2f}'.format(states[n][0], states[n][1]))
        a1.imshow(states[n][3], cmap='gray')
        a1.set_aspect('equal')
        a1.axis('off')

        a2 = plt.axes([xa + 0.005, ya - img_size / 2.0, img_size, img_size])
        a2.imshow(states[n][2], cmap='gray')
        a2.set_aspect('equal')
        a2.axis('off')

      plt.show()
