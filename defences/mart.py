import torch.nn as nn
import torch
import torch.nn.functional as F
from attacks import pgd_attack

def mart_loss(model, x_natural, y, optimizer, epsilon=0.031, alpha=0.007,  iters=10, beta=6.0):
  
  criterion_kl = nn.KLDivLoss(reduction='none')
  
  model.eval()
  x_adv = pgd_attack(model, x_natural, y, epsilon, alpha, random_start=True, norm="l_inf", iters=iters)
  
  model.train()
  optimizer.zero_grad()
  
  logits_natural = model(x_natural)
  logits_adv = model(x_adv)
  
  loss_natural = F.cross_entropy(logits_natural, y)
  
  probs_natural = F.softmax(logits_natural, dim=1)
  probs_adv = F.log_softmax(logits_adv, dim=1)
  
  loss_robust = criterion_kl(probs_adv, probs_natural).sum(dim=1)
      
  pred_natural = logits_natural.argmax(dim=1)
  boosting = torch.where(pred_natural == y, torch.zeros_like(loss_robust),beta * loss_robust)
  
  loss = loss_natural + boosting.mean()
  loss.backward()
  optimizer.step()
  
  return loss.item()