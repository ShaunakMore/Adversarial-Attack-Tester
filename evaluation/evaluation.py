import torch
from attacks import pgd_attack
from attacks import fgsm_attack
from attacks import cw_binary_search

def evaluate_clean(model, test_loader, device):
  
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            outputs = model(x_batch)
            _, predicted = torch.max(outputs, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
    
    return 100 * correct / total

def evaluate_pgd(model, test_loader, device, epsilon=0.031, alpha=0.007, iters=20):
    
    model.eval()
    correct = 0
    total = 0
    
    for x_batch, y_batch in test_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        
        x_adv = pgd_attack(model, x_batch, y_batch, epsilon, alpha, 
                          random_start=True, norm="l_inf", iters=iters)
        
        with torch.no_grad():
            outputs = model(x_adv)
            _, predicted = torch.max(outputs, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
    
    return 100 * correct / total

def evaluate_fgsm(model, test_loader, device, epsilon=0.031):
   
    model.eval()
    correct = 0
    total = 0
    
    for x_batch, y_batch in test_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        
        # Generate adversarial examples
        x_adv = fgsm_attack(model, x_batch, y_batch, epsilon, device)
        
        with torch.no_grad():
            outputs = model(x_adv)
            _, predicted = torch.max(outputs, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
    
    return 100 * correct / total

def evaluate_cw(model, test_loader, device, num_samples=100):
    
    model.eval()
    correct = 0
    total = 0
    
    for x_batch, y_batch in test_loader:
        if total >= num_samples:
            break
        
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        
        for x, y in zip(x_batch, y_batch):
            if total >= num_samples:
                break
            
            # Use C&W attack
            target = (y.item() + 1) % 10  
            x_adv = cw_binary_search(model, x, target, binary_search_steps=5, max_iters=100)
            
            with torch.no_grad():
                output = model(x_adv.unsqueeze(0))
                pred = torch.argmax(output)
                correct += (pred == y).item()
                total += 1
    
    return 100 * correct / total