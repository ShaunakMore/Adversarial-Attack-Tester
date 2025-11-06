import torch.nn as nn
import torch
import torch.nn.functional as F
from defences import adversarial_training_pgd, adversarial_training_fgsm
from defences import mart_loss
from defences import trades_loss
from evaluation import evaluate_clean, evaluate_pgd

def train_robust_model(model, train_loader, test_loader, num_epochs=100, method='trades', device='cuda'):
  
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 75], gamma=0.1)
    
    model = model.to(device)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        loss = 0
        
        for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            
            if method == 'pgd':
              loss = adversarial_training_pgd(model, x_batch, y_batch, optimizer,epsilon=0.031, alpha=0.007, iters=10)
            
            elif method == 'fgsm':
              loss = adversarial_training_fgsm(model, x_batch, y_batch, optimizer, epsilon=0.031)
            
            elif method == 'trades':
              loss, loss_nat, loss_rob = trades_loss(model, x_batch, y_batch, optimizer,epsilon=0.031, alpha=0.007, iters=10, beta=6.0)
            
            elif method == 'mart':
              loss = mart_loss(model, x_batch, y_batch, optimizer,epsilon=0.031, alpha=0.007, iters=10, beta=6.0)
            
            elif method == 'mixed':
                if batch_idx % 2 == 0:
                    optimizer.zero_grad()
                    output = model(x_batch)
                    loss = F.cross_entropy(output, y_batch)
                    loss.backward()
                    optimizer.step()
                    loss = loss.item()
                else:
                    loss = adversarial_training_pgd(model, x_batch, y_batch, optimizer,epsilon=0.031, alpha=0.007, iters=10)
            
            total_loss += loss
            
            if batch_idx % 100 == 0:
              print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss:.4f}')
        
        scheduler.step()
        
        # Evaluate
        if epoch % 5 == 0:
            clean_acc = evaluate_clean(model, test_loader, device)
            pgd_acc = evaluate_pgd(model, test_loader, device)
            print(f'\nEpoch {epoch}:')
            print(f'Clean Accuracy: {clean_acc:.2f}%')
            print(f'PGD Accuracy: {pgd_acc:.2f}%\n')
    
    return model