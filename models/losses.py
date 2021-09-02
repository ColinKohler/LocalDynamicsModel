import torch
import torch.nn.functional as F

def smoothnessLoss(depth_img):
  loss = torch.sum(torch.abs(depth_img[:,:,:,:-1] - depth_img[:,:,:,1:]))+ \
         torch.sum(torch.abs(depth_img[:,:,:-1,:] - depth_img[:,:,1:,:]))

  return loss

def reconLoss(x, y, smoothness_weight=0):
  return F.mse_loss(x, y, reduction='mean') + smoothness_weight * smoothnessLoss(x)

def klLoss(mu, log_var):
  return -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

def betaKlLoss(beta, mu, log_var):
  return beta * klLoss(mu, log_var)

def maskedNLL(x, y, mask, weight=None):
  try:
    loss = F.nll_loss(x, y, reduction='none', weight=weight)
    return torch.mean(mask * torch.mean(loss.view(mask.size(0), -1), axis=1))
  except Exception as e:
    print(e)
    breakpoint()

def maskedImageNLL(x, y, mask, weight_map=None):
  try:
    loss = F.nll_loss(x, y, reduction='none')
    if weight_map is not None: loss = weight_map * loss
    return torch.mean(mask * torch.mean(loss.view(mask.size(0), -1), axis=1))
  except Exception as e:
    print(e)
    breakpoint()

def maskedMSE(x, y, mask):
  return torch.mean(mask * torch.mean(F.mse_loss(x, y, reduction='none').view(mask.size(0), -1), axis=1))

def maskedHuberLoss(x, y, mask):
  return torch.mean(mask * torch.mean(F.smooth_l1_loss(x, y, reduction='none').view(mask.size(0), -1), axis=1))
