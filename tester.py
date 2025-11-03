import numpy as np
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