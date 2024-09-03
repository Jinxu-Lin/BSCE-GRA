'''
This module contains methods for training models with different loss functions.
'''

import torch
from torch.nn import functional as F
from torch import nn

from Losses.loss import cross_entropy, cross_entropy_exp, cross_entropy_weight_bs
from Losses.loss import focal_loss, focal_loss_gra, focal_loss_adaptive, focal_loss_adaptive_gra
from Losses.loss import dual_focal_loss, dual_focal_loss_gra
from Losses.loss import mmce, mmce_weighted, mmce_gra
from Losses.loss import brier_score, brier_score_exp, brier_score_exp_no_clipping, brier_score_exp_no_minus, brier_score_exp_pure
from Losses.loss import bsce, bsce_gra, bsce_adaptive_gra, tlbs
from Losses.loss import ece_loss, dece

loss_function_dict = {
    'cross_entropy': cross_entropy,
    'cross_entropy_exp': cross_entropy_exp,
    'cross_entropy_weight_bs': cross_entropy_weight_bs,
    'brier_score_exp_no_clipping': brier_score_exp_no_clipping,
    'brier_score_exp_no_minus': brier_score_exp_no_minus,
    'brier_score_exp_pure': brier_score_exp_pure,
    'focal_loss': focal_loss,
    'focal_loss_gra': focal_loss_gra,
    'focal_loss_adaptive': focal_loss_adaptive,
    'focal_loss_adaptive_gra': focal_loss_adaptive_gra,
    'dual_focal_loss': dual_focal_loss,
    'dual_focal_loss_gra': dual_focal_loss_gra,
    'mmce': mmce,
    'mmce_gra': mmce_gra,
    'mmce_weighted': mmce_weighted,
    'brier_score': brier_score,
    'bsce': bsce,
    'brier_score_exp': brier_score_exp,
    'bsce_gra': bsce_gra,
    'bsce_adaptive_gra': bsce_adaptive_gra,
    'ece_loss': ece_loss,
    'tlbs': tlbs,
    'dece': dece,
}


def train_single_epoch(epoch,
                       model,
                       train_loader,
                       optimizer,
                       device,
                       loss_function='cross_entropy',
                       gamma=1.0,
                       lamda=1.0,
                       n_bins=5,
                       bsce_norm=1,
                       size_average=False,
                       temperature=1.0,
                       loss_mean=False):
    '''
    Util method for training a model for a single epoch.
    '''
    log_interval = 10
    model.train()
    train_loss = 0
    num_samples = 0
    for batch_idx, (data, labels) in enumerate(train_loader):
        data = data.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(data)
        if ('mmce' in loss_function):
            loss = (len(data) * loss_function_dict[loss_function](logits, labels, gamma=gamma, lamda=lamda, device=device))
        else:
            loss = loss_function_dict[loss_function](logits, labels, temperature=temperature, gamma=gamma, lamda=lamda, n_bins=n_bins, bsce_norm=bsce_norm, size_average=size_average, device=device)

        if loss_mean:
            loss = loss / len(data)

        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), 2)
        train_loss += loss.item()
        optimizer.step()
        num_samples += len(data)

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader) * len(data),
                100. * batch_idx / len(train_loader),
                loss.item()))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / num_samples))
    return train_loss / num_samples



def test_single_epoch(epoch,
                      model,
                      test_val_loader,
                      device,
                      loss_function='cross_entropy',
                      gamma=1.0,
                      lamda=1.0,
                      n_bins=5,
                      bsce_norm=1,
                      size_average=False,
                      temperature=1.0,
                      ):
    '''
    Util method for testing a model for a single epoch.
    '''
    model.eval()
    loss = 0
    num_samples = 0
    with torch.no_grad():
        for i, (data, labels) in enumerate(test_val_loader):
            data = data.to(device)
            labels = labels.to(device)

            logits = model(data)
            if ('mmce' in loss_function):
                loss += (len(data) * loss_function_dict[loss_function](logits, labels, gamma=gamma, lamda=lamda, device=device).item())
            else:
                loss += loss_function_dict[loss_function](logits, labels, temperature=temperature, gamma=gamma, lamda=lamda, n_bins=n_bins, bsce_norm=bsce_norm, size_average=size_average, device=device).item()
            num_samples += len(data)

    print('======> Test set loss: {:.4f}'.format(
        loss / num_samples))
    return loss / num_samples