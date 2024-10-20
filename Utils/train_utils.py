'''
This module contains methods for training models with different loss functions.
'''

import torch
from torch.nn import functional as F
from torch import nn
from Utils.eval_utils import evaluate_dataset

def train_single_epoch(args,
                       epoch,
                       model,
                       train_loader,
                       val_loader,
                       optimizer,
                       device,
                       loss_function,
                       num_labels,
                       calibrator,
                    ):
    '''
    Util method for training a model for a single epoch.
    '''
    log_interval = 10
    model.train()
    train_loss = 0
    num_samples = 0
    predictions_list = []
    confidence_list = []
    labels_list = []
    for batch_idx, (data, labels) in enumerate(train_loader):
        data = data.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(data)

        if batch_idx == 0:
            fulldataset_logits = logits
        else:
            fulldataset_logits = torch.cat((fulldataset_logits, logits), dim=0)

        # Compute loss
        if args.loss_function == "consistency":
            calibrated_probability = calibrator.calibrate(logits)
            loss = loss_function(logits, labels, calibrated_probability)
        elif args.loss_function in ('mmce', 'mmce_gra', 'mmce_weighted'):
            loss = (len(data) * loss_function(logits, labels))
        elif args.loss_function == "ece_loss":
            loss = loss_function(logits, labels, epoch)
        else:
            loss = loss_function(logits, labels)

        # Compute confidence values
        log_softmax = F.log_softmax(logits, dim=1)
        log_confidence, predictions = torch.max(log_softmax, dim=1)
        confidence = log_confidence.exp()  

        if args.loss_mean:
            loss = loss / len(data)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
        train_loss += loss.item()
        optimizer.step()
        
        num_samples += len(data)

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader) * len(data),
                100. * batch_idx / len(train_loader),
                loss.item()))
        
        if args.loss_function == "adafocal" and args.update_gamma_every == -1 and batch_idx == len(train_loader)-1:
            print("Gamma updated after the end of epoch.")
            (val_loss, val_confusion_matrix, val_acc, val_ece, val_bin_dict,
            val_adaece, val_adabin_dict, val_mce, val_classwise_ece) = evaluate_dataset(model, val_loader, device, num_bins=args.num_bins, num_labels=num_labels)
            loss_function.update_bin_stats(val_adabin_dict)
        elif args.loss_function == "adafocal" and args.update_gamma_every > 0 and batch_idx > 0 and batch_idx % args.update_gamma_every == 0:
            print("Gamma updated after batch:", batch_idx)
            (val_loss, val_confusion_matrix, val_acc, val_ece, val_bin_dict,
            val_adaece, val_adabin_dict, val_mce, val_classwise_ece) = evaluate_dataset(model, val_loader, device, num_bins=args.num_bins, num_labels=num_labels)
            loss_function.update_bin_stats(val_adabin_dict)

        # Collect predictions, confidence values, and labels over the entire dataset
        predictions_list.extend(predictions.cpu().numpy().tolist())
        confidence_list.extend(confidence.detach().cpu().numpy().tolist())
        labels_list.extend(labels.cpu().numpy().tolist())
            
    train_loss = train_loss/num_samples
    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss))
    return train_loss, loss_function, labels_list, fulldataset_logits, predictions_list, confidence_list


def train_single_epoch_warmup(args,
                       epoch,
                       model,
                       train_loader,
                       val_loader,
                       optimizer,
                       device,
                       loss_function,
                       num_labels,
                    ):
    '''
    Util method for training a model for a single epoch.
    '''
    log_interval = 10
    model.train()
    train_loss = 0
    num_samples = 0
    predictions_list = []
    confidence_list = []
    labels_list = []
    for batch_idx, (data, labels) in enumerate(train_loader):
        data = data.to(device)
        labels = labels.to(device)


        optimizer.zero_grad()

        logits = model(data)

        if batch_idx == 0:
            fulldataset_logits = logits
        else:
            fulldataset_logits = torch.cat((fulldataset_logits, logits), dim=0)
        
        loss = F.cross_entropy(logits, labels, reduction='sum')

        # Compute confidence values
        log_softmax = F.log_softmax(logits, dim=1)
        log_confidence, predictions = torch.max(log_softmax, dim=1)
        confidence = log_confidence.exp()    

        if args.loss_mean:
            loss = loss / len(data)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
        train_loss += loss.item()
        optimizer.step()
        
        num_samples += len(data)

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader) * len(data),
                100. * batch_idx / len(train_loader),
                loss.item()))
        
        if args.loss_function == "adafocal" and args.update_gamma_every == -1 and batch_idx == len(train_loader)-1:
            print("Gamma updated after the end of epoch.")
            (val_loss, val_confusion_matrix, val_acc, val_ece, val_bin_dict,
            val_adaece, val_adabin_dict, val_mce, val_classwise_ece) = evaluate_dataset(model, val_loader, device, num_bins=args.num_bins, num_labels=num_labels)
            loss_function.update_bin_stats(val_adabin_dict)
        elif args.loss_function == "adafocal" and args.update_gamma_every > 0 and batch_idx > 0 and batch_idx % args.update_gamma_every == 0:
            print("Gamma updated after batch:", batch_idx)
            (val_loss, val_confusion_matrix, val_acc, val_ece, val_bin_dict,
            val_adaece, val_adabin_dict, val_mce, val_classwise_ece) = evaluate_dataset(model, val_loader, device, num_bins=args.num_bins, num_labels=num_labels)
            loss_function.update_bin_stats(val_adabin_dict)
        
        # Collect predictions, confidence values, and labels over the entire dataset
        predictions_list.extend(predictions.cpu().numpy().tolist())
        confidence_list.extend(confidence.detach().cpu().numpy().tolist())
        labels_list.extend(labels.cpu().numpy().tolist())
            
    train_loss = train_loss/num_samples
    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss))
    return train_loss, loss_function, labels_list, fulldataset_logits, predictions_list, confidence_list


def set_temperature(model):
    temperature = 1.0
    return temperature