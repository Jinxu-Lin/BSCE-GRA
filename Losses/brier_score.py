'''
Implementation of Brier Score.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from scipy.special import lambertw
import numpy as np


def get_gamma(p=0.2):
    '''
    Get the gamma for a given pt where the function g(p, gamma) = 1
    '''
    y = ((1-p)**(1-(1-p)/(p*np.log(p)))/(p*np.log(p)))*np.log(1-p)
    gamma_complex = (1-p)/(p*np.log(p)) + lambertw(-y + 1e-12, k=-1)/np.log(1-p)
    gamma = np.real(gamma_complex) #gamma for which p_t > p results in g(p_t,gamma)<1
    return gamma


ps = [0.2, 0.5]
gammas = [5.0, 3.0]
i = 0
gamma_dic = {}
for p in ps:
    gamma_dic[p] = gammas[i]
    i += 1


class BrierScore(nn.Module):
    def __init__(self):
        super(BrierScore, self).__init__()

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)
        target_one_hot = torch.FloatTensor(input.shape).to(target.get_device())
        target_one_hot.zero_()
        target_one_hot.scatter_(1, target, 1)

        pt = F.softmax(input)
        squared_diff = (target_one_hot - pt) ** 2

        loss = torch.sum(squared_diff)
        return loss
    

class BrierScoreExp(nn.Module):
    def __init__(self, temperature=1.0):
        super(BrierScoreExp, self).__init__()
        self.temperature = temperature

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)
        
        target_one_hot = torch.FloatTensor(input.shape).to(target.get_device())
        target_one_hot.zero_()
        target_one_hot.scatter_(1, target, 1)

        pt = F.softmax(input)
        squared_diff = (target_one_hot - pt) ** 2
        squared_diff = squared_diff.sum(dim=1)
    
        with torch.no_grad():
            weight = torch.exp(
                torch.clamp(squared_diff.detach(), min=0, max=self.temperature) / (self.temperature + 1)
            ) - 1

        loss = torch.sum(squared_diff*weight)

        return loss
    

class BrierScoreExpNoClipping(nn.Module):
    def __init__(self, temperature=1.0):
        super(BrierScoreExpNoClipping, self).__init__()
        self.temperature = temperature

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)
        
        target_one_hot = torch.FloatTensor(input.shape).to(target.get_device())
        target_one_hot.zero_()
        target_one_hot.scatter_(1, target, 1)

        pt = F.softmax(input)
        squared_diff = (target_one_hot - pt) ** 2
            
        with torch.no_grad():
            weight = torch.exp(
                torch.clamp(squared_diff.detach(), min=0, max=self.temperature)
            ) - 1

        loss = torch.sum(squared_diff*weight)

        return loss
    

class BSCELoss(nn.Module):
    def __init__(self, gamma=0, norm=1, size_average=False):
        super(BSCELoss, self).__init__()
        self.gamma = gamma
        self.norm = norm
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = F.softmax(input)

        target_one_hot = torch.FloatTensor(input.shape).to(target.get_device())
        target_one_hot.zero_()
        target_one_hot.scatter_(1, target, 1)
        # with torch.no_grad():
        diff = torch.norm(target_one_hot - pt, p=self.norm, dim=1) ** self.gamma
        loss = -1 * diff * logpt

        if self.size_average: return loss.mean()
        else: return loss.sum()


class BSCELossGra(nn.Module):
    def __init__(self, gamma=0, norm=1, size_average=False):
        super(BSCELossGra, self).__init__()
        self.gamma = gamma
        self.norm = norm
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = F.softmax(input)

        target_one_hot = torch.FloatTensor(input.shape).to(target.get_device())
        target_one_hot.zero_()
        target_one_hot.scatter_(1, target, 1)
        with torch.no_grad():
            diff = torch.norm(target_one_hot - pt, p=self.norm, dim=1) ** self.gamma
        loss = -1 * diff * logpt

        if self.size_average: return loss.mean()
        else: return loss.sum()


class BSCELossAdaptiveGra(nn.Module):
    def __init__(self, gamma=0, norm=1, size_average=False, device=None):
        super(BSCELossAdaptiveGra, self).__init__()
        self.gamma = gamma
        self.norm = norm
        self.size_average = size_average
        self.device = device

    def get_gamma_list(self, pt):
        gamma_list = []
        batch_size = pt.shape[0]
        for i in range(batch_size):
            pt_sample = pt[i].item()
            if (pt_sample >= 0.5):
                gamma_list.append(self.gamma)
                continue
            # Choosing the gamma for the sample
            for key in sorted(gamma_dic.keys()):
                if pt_sample < key:
                    gamma_list.append(gamma_dic[key])
                    break
        return torch.tensor(gamma_list).to(self.device)

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = F.softmax(input)

        target_one_hot = torch.FloatTensor(input.shape).to(target.get_device())
        target_one_hot.zero_()
        target_one_hot.scatter_(1, target, 1)
        p = logpt.exp()
        gamma = self.get_gamma_list(p)
        with torch.no_grad():
            diff = torch.norm(target_one_hot - pt, p=self.norm, dim=1) ** gamma
        loss = -1 * diff * logpt

        if self.size_average: return loss.mean()
        else: return loss.sum()


class TLBSLoss(nn.Module):
    def __init__(self, gamma=0, size_average=False, device='cuda'):
        super(TLBSLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average
        self.device = device

    def forward(self, input, target):

        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C

        target = target.view(-1) #For CIFAR-10 and CIFAR-100, target.shape is [N] to begin with

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target.view(-1, 1))
        logpt = logpt.view(-1)

        predicted_probs = F.softmax(input, dim=1)
        predicted_probs, pred_labels = torch.max(predicted_probs, 1)
        correct_mask = torch.where(torch.eq(pred_labels, target),
                          torch.ones(pred_labels.shape).to(self.device),
                          torch.zeros(pred_labels.shape).to(self.device))

        with torch.no_grad():
            c_minus_r = (correct_mask - predicted_probs).abs() ** self.gamma
        
        loss = -1 * c_minus_r * logpt

        if self.size_average: return loss.mean()
        else: return loss.sum()
        