import torch.nn as nn
import torch

def pgd_attack(model, x, y, epsilon=0.03, alpha=0.007, random_start=True, norm="l_inf", iters=40):

    loss_fn = nn.CrossEntropyLoss()
    
    single_sample = False
    if x.dim() == 3:  
        x = x.unsqueeze(0)
        y = y.unsqueeze(0) if y.dim() == 0 else y
        single_sample = True
    
    if norm == "l_inf":
        if random_start:
            x_adv = x + torch.empty_like(x).uniform_(-epsilon, epsilon)
            x_adv = torch.clamp(x_adv, 0, 1)
        else:
            x_adv = x.clone()
        
        for i in range(iters):
            x_adv.requires_grad_()
            
            output = model(x_adv)
            loss = loss_fn(output, y)
            
            model.zero_grad()
            loss.backward()
            
            with torch.no_grad():
                sign = x_adv.grad.detach().sign()
                x_adv = x_adv + alpha * sign
                
                perturbation = x_adv - x
                perturbation = torch.clamp(perturbation, -epsilon, epsilon)
                x_adv = x + perturbation
                x_adv = torch.clamp(x_adv, 0, 1)
            
            x_adv = x_adv.detach()
        
        result = x_adv.detach()
    
    elif norm.lower() in ("l2", "l_2", "l_2norm"): 
        if random_start:
            delta = torch.randn_like(x)
            delta_norm = torch.norm(delta.view(delta.shape[0], -1), dim=1, keepdim=True)
            delta = delta / (delta_norm.view(-1, 1, 1, 1) + 1e-8) * epsilon
            x_adv = torch.clamp(x + delta, 0, 1)
        else:
            x_adv = x.clone()
        
        for i in range(iters):
            x_adv.requires_grad_()
            loss = loss_fn(model(x_adv), y)
            
            model.zero_grad()
            loss.backward()
            
            grad = x_adv.grad.detach()
            grad_norm = torch.norm(grad.view(grad.shape[0], -1), dim=1, keepdim=True)
            grad_normalised = grad / (grad_norm.view(-1, 1, 1, 1) + 1e-8)
            
            with torch.no_grad():
                x_adv = x_adv + alpha * grad_normalised
                
                delta = x_adv - x
                delta_norm = torch.norm(delta.view(delta.shape[0], -1), dim=1, keepdim=True)
                factor = torch.min(torch.ones_like(delta_norm), epsilon / (delta_norm + 1e-8))
                delta = delta * factor.view(-1, 1, 1, 1)
                
                x_adv = torch.clamp(x + delta, 0, 1)
            
            x_adv = x_adv.detach()
        
        result = x_adv.detach()

        if single_sample:
            result = result.squeeze(0)
        
        return result
