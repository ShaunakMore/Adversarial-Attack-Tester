import torch.nn as nn
import torch
import torch.nn.functional as F
from attacks import pgd_attack

def trades_loss(model, x_natural, y, optimizer, epsilon=0.031, alpha=0.007, iters=10, beta=6.0):

    criterion_kl = nn.KLDivLoss(reduction='sum')
    
    model.eval()
    x_adv = pgd_attack(model, x_natural, y, epsilon, alpha, random_start=True, norm="l_inf", iters=iters)
    
    model.train()
    optimizer.zero_grad()
    
    logits_natural = model(x_natural)
    logits_adv = model(x_adv)
    
    loss_natural = F.cross_entropy(logits_natural, y)
    
    loss_robust = (1.0 / len(x_natural)) * criterion_kl(
        F.log_softmax(logits_adv, dim=1),
        F.softmax(logits_natural, dim=1)
    )
    
    loss = loss_natural + beta * loss_robust
    loss.backward()
    optimizer.step()
    
    return loss.item(), loss_natural.item(), loss_robust.item()