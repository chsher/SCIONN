import torch
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

import pdb

PRINT_STMT = 'Epoch: {0:3d}, Batch: {1:3d}, {2:5} BCE1: {3:7.4f}, BCE2: {4:7.4f}, Omega: {5:7.4f}, {8}: {6:7.4f}, Frac: {7:9.6f}'

def run_training_loop(e, train_loader, net, loss_fn, optimizer, device, lamb=None, temp=None, gumbel=False, adv=False, blabel='Train', verbose=True):
    
    net.train()
    net.to(device)

    for b, (input, label, base) in enumerate(train_loader):
        input, label, base = input.to(device), label.to(device), base.to(device)
        
        outputs = net(input)

        if gumbel:
            if adv:
                output_keep, keep, output_adv = outputs

                loss_keep = loss_fn(output_keep, label)
                loss_adv = loss_fn(output_adv, label)

                lamb_keep = 1.0 / torch.linalg.norm(loss_keep.detach())
                lamb_adv = 1.0 / torch.linalg.norm(loss_adv.detach())

                bceloss = lamb_keep * loss_keep - lamb_adv * loss_adv

                losses = [loss_keep, loss_adv]
                
            else:
                output_keep, keep = outputs

                bceloss = loss_fn(output_keep, label)

                losses = [bceloss, np.nan]
            
            znorm = torch.sum(torch.max(keep[:, :, -1], base))
            omega = (lamb * znorm) / input.shape[0]
            losses.append(omega)

            loss = bceloss + omega
            
            frac_tiles = znorm / (input.shape[0] * input.shape[1])

        else:
            output_keep = outputs

            loss = loss_fn(output_keep, label)
            losses = [loss, np.nan, np.nan]

            frac_tiles = np.nan

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        try:     
            if label.shape[1] == 1:
                y_prob = torch.sigmoid(output_keep.detach())
                auc = roc_auc_score(label.cpu().squeeze(-1).numpy(), y_prob.cpu().squeeze(-1).numpy())
            else:
                y_prob = F.softmax(output_keep.detach(), dim=-1)
                auc = np.mean(np.abs(label.cpu().squeeze(-1).numpy() - y_prob.cpu().squeeze(-1).numpy()))
        except:
            auc = np.nan

        if verbose:
            metric = 'AUC' if label.shape[1] == 1 else 'MAE'
            print(PRINT_STMT.format(e, b, blabel, *losses, auc, frac_tiles, metric))

def run_validation_loop(e, val_loader, net, loss_fn, device, lamb=None, temp=None, gumbel=True, adv=False, blabel='Val', verbose=True):
    net.eval()
    net.to(device)

    total_znorm, total_tiles, total_loss = 0.0, 0.0, 0.0
    y_tracker, y_prob_tracker = np.array([]), np.array([])

    for b, (input, label, base) in enumerate(val_loader):
        
        input, label, base = input.to(device), label.to(device), base.to(device)
        
        outputs = net(input)

        if gumbel:
            if adv:
                output_keep, keep, output_adv = outputs

                loss_keep = loss_fn(output_keep, label)
                loss_adv = loss_fn(output_adv, label)

                lamb_keep = 1.0 / torch.linalg.norm(loss_keep.detach())
                lamb_adv = 1.0 / torch.linalg.norm(loss_adv.detach())

                bceloss = lamb_keep * loss_keep - lamb_adv * loss_adv

                losses = [loss_keep, loss_adv]
                
            else:
                output_keep, keep = outputs

                bceloss = loss_fn(output_keep, label)

                losses = [bceloss, np.nan]
            
            znorm = torch.sum(torch.max(keep[:, :, -1], base))
            omega = (lamb * znorm) / input.shape[0]
            losses.append(omega)

            loss = bceloss + omega
            
            frac_tiles = znorm / (input.shape[0] * input.shape[1])
            total_znorm += znorm
            total_tiles += input.shape[0] * input.shape[1]

        else:
            output_keep = outputs

            loss = loss_fn(output_keep, label)
            losses = [loss, np.nan, np.nan]

            frac_tiles = np.nan

        total_loss += loss.detach().cpu().numpy()
        
        if label.shape[1] == 1:
            y_prob = torch.sigmoid(output_keep.detach())
            y_prob_tracker = np.concatenate((y_prob_tracker, y_prob.cpu().squeeze(-1).numpy()))
            y_tracker = np.concatenate((y_tracker, label.cpu().squeeze(-1).numpy()))
        else:
            y_prob = F.softmax(output_keep.detach(), dim=-1)
            y_prob_tracker = np.concatenate((y_prob_tracker, y_prob.cpu().squeeze(-1).numpy().flatten()))
            y_tracker = np.concatenate((y_tracker, label.cpu().squeeze(-1).numpy().flatten()))

        try:     
            if label.shape[1] == 1:
                auc = roc_auc_score(label.cpu().squeeze(-1).numpy(), y_prob.cpu().squeeze(-1).numpy())
            else:
                auc = np.mean(np.abs(label.cpu().squeeze(-1).numpy() - y_prob.cpu().squeeze(-1).numpy()))
        except:
            auc = np.nan

        if verbose:
            metric = 'AUC' if label.shape[1] == 1 else 'MAE'
            print(PRINT_STMT.format(e, b, blabel, *losses, auc, frac_tiles, metric))

    try:
        total_frac = total_znorm / total_tiles
        total_frac = total_frac.item()
    except:
        total_frac = np.nan

    if label.shape[1] == 1:
        return total_loss, roc_auc_score(y_tracker, y_prob_tracker), total_frac
    else:
        return total_loss, np.mean(np.abs(y_tracker - y_prob_tracker)), total_frac