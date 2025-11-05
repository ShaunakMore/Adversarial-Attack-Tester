import numpy as np
import torch.nn as nn
import torch

def fgsm_attack(model,image,label,epsilon=0.03, device = None):
  
  if device is None:
    device = next(model.parameters()).device
  
  image = image.clone().detach().to(device)
  label = label.to(device)
  image.requires_grad = True
  
  loss_fn = nn.CrossEntropyLoss()
  output = model(image)
  loss = loss_fn(output,label)
  
  model.zero_grad()
  if image.grad is not None:
      image.grad.detach_()
      image.grad.zero_()
  loss.backward()
  gradient = image.grad
  
  perturbed = epsilon * gradient.sign()
  
  perturbed_image = image + perturbed
  
  perturbed_image = torch.clamp(perturbed_image,0,1)
  
  return perturbed_image.detach()

def fgsm_targeted(model, image, target_class, epsilon=0.03):
 
    image.requires_grad = True
    
    output = model(image)
    
    loss_fn = nn.CrossEntropyLoss()
    
    loss = loss_fn(output, target_class)
    
    model.zero_grad()
    loss.backward()
    
    perturbation = -epsilon * image.grad.sign()
    
    perturbed = torch.clamp(image + perturbation, 0, 1)
    return perturbed.detach()
 
def pgd_attack(model,x,y,epsilon = 0.03,alpha = 0.007,random_start = True, norm="l_inf",iters = 40):
  
  loss_fn = nn.CrossEntropyLoss()
  
  if (norm == "l_inf"):
    if random_start:
     x_adv = x + torch.empty_like(x).uniform_(-epsilon,epsilon)
     x_adv = torch.clamp(x_adv,0,1)
    else:
      x_adv = x.clone()
      
    for i in range(iters):
      x_adv.requires_grad_()
    
      output = model(x_adv)
      loss = loss_fn(output,y)
    
      model.zero_grad()
      loss.backward()
    
      with torch.no_grad():
        sign = x_adv.grad.detach().sign()
      
        x_adv = x_adv + alpha * sign
        perturbtion  = x_adv - x
        perturbtion = torch.clamp(perturbtion, -epsilon, epsilon)
        x_adv = x + perturbtion
      
        x_adv = torch.clamp(x_adv,0,1)
      x_adv = x_adv.detach()
    return x_adv.detach() 
  
  elif norm.lower() in ("l2", "l_2", "l_2norm"): 
    x_adv = x.clone().detach()
    
    for i in range(iters):
      x_adv.requires_grad_()
      loss= loss_fn(model(x_adv),y)
      
      model.zero_grad()
      loss.backward()
      
      grad = x_adv.grad.detach()
      grad_norm = torch.norm(grad.view(grad.shape[0],-1),dim =1, keepdim=True)
      grad_normalised = grad / (grad_norm.view(-1,1,1,1) + 1e-8)
      
      with torch.no_grad():
        x_adv = x_adv + alpha * grad_normalised
        
        delta = x_adv - x
        delta_norm = torch.norm(delta.view(delta.shape[0],-1), dim = 1,keepdim=True)
        factor = torch.min(torch.ones_like(delta_norm),epsilon/( + 1e-8))
        delta = delta * factor.view(-1,1,1,1)
        
        x_adv = torch.clamp(x+ delta,0,1)
      x_adv = x_adv.detach()
    return x_adv.detach()
  
    
    