import torch.nn as nn
import torch
from attacks import fgsm_attack
from attacks import pgd_attack

def adversarial_training_pgd(model, x_batch, y_batch, optimiser, epsilon = 0.03,alpha = 0.007,random_start = True, norm="l_inf",iters = 40 ):
  
  model.eval()
  x_adv_batch = pgd_attack(model,x_batch,y_batch,epsilon,alpha,random_start, norm,iters)
  
  model.train()
  optimiser.zero_grad()
  loss_fn = nn.CrossEntropyLoss()
  outputs = model(x_adv_batch)
  loss = loss_fn(outputs,y_batch)
  
  loss.backward()
  optimiser.step()
  
  return loss.item()

def adversarial_training_fgsm(model, x_batch, y_batch, optimizer, epsilon=0.03):

    loss_fn = nn.CrossEntropyLoss()

    model.eval()
    x_adv = fgsm_attack(model, x_batch, y_batch, epsilon)
    
    model.train()
    optimizer.zero_grad()
    
    output = model(x_adv)
    loss = loss_fn(output, y_batch)
    
    loss.backward()
    optimizer.step()
    
    return loss.item()