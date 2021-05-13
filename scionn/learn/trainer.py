import torch
import numpy as np
from sklearn.metrics import roc_auc_score

PRINT_STMT = 'Epoch: {0:3d}, Batch: {1:3d}, {2:5} BCE1: {3:7.4f}, BCE2: {4:7.4f}, Omega: {5:7.4f}, AUC: {6:7.4f}, Frac: {7:9.6f}'

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
             
        auc = roc_auc_score(label.cpu().numpy(), output_keep.detach().cpu().numpy())

        if verbose:
            print(PRINT_STMT.format(e, b, blabel, *losses, 0.0, 0.0, auc, frac_tiles))

def run_validation_loop(e, val_loader, net, loss_fn, device, lamb=None, temp=None, gumbel=True, adv=False, blabel='Val', verbose=True):
    net.eval()
    net.to(device)

    total_loss = 0.0
    for b, (input, label, base) in enumerate(val_loader):
        input, label, base = input.to(device), label.to(device), base.to(device)
        
        output = gen(input)

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

        total_loss += loss.detach().cpu().numpy()

        auc = roc_auc_score(label.cpu().numpy(), output_keep.detach().cpu().numpy())

        if verbose:
            print(PRINT_STMT.format(e, b, blabel, *losses, 0.0, 0.0, auc, frac_tiles))

    return total_loss, auc, frac_tiles.item()