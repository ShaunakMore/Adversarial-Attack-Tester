import torch.nn as nn
import torch

def cw_l2_attack(model, x, target, c=1.0, kappa=0, max_iters=1000, learning_rate=0.001):
    w = torch.arctanh(2 * x - 1)
    w = w.detach().requires_grad_()
    
    optimizer = torch.optim.Adam([w], lr=learning_rate)
    
    best_adv = x.clone()
    best_l2 = float('inf')
    
    for iteration in range(max_iters):
        x_adv = 0.5 * (torch.tanh(w) + 1)
        
        output = model(x_adv.unsqueeze(0))
        logits = output[0]
        
        target_logit = logits[target]
        
        other_logits = logits.clone()
        other_logits[target] = -float('inf')
        max_other_logit = torch.max(other_logits)
        
        f_loss = torch.clamp(max_other_logit - target_logit + kappa, min=0)
        
        l2_dist = torch.norm((x_adv - x).flatten(), p=2)
        
        total_loss = l2_dist + c * f_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        if f_loss.item() == 0 and l2_dist.item() < best_l2:
            best_l2 = l2_dist.item()
            best_adv = x_adv.clone().detach()
        
        if iteration % 100 == 0:
            print(f"\nIter {iteration}: L2 = {l2_dist.item():.4f}, f = {f_loss.item():.4f}")   
    
    return best_adv

def cw_binary_search(model, x, target, binary_search_steps=9, max_iters=1000):
    c_lower = 0 
    c_upper = 1e10
    
    best_adv = x.clone()
    best_l2 = float('inf')
    
    for search_step in range(binary_search_steps):
        c = (c_lower + c_upper) / 2
        print(f"\nBinary search step {search_step}, c = {c}")
        
        adv = cw_l2_attack(model, x, target, c, max_iters=max_iters)
        
        pred = torch.argmax(model(adv.unsqueeze(0)))
        l2_dist = torch.norm((adv - x).flatten(), p=2).item()
        
        if pred == target:
            c_upper = c
            if l2_dist < best_l2:
                best_l2 = l2_dist
                best_adv = adv
        else:
            c_lower = c
    
    return best_adv