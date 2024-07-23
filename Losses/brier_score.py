'''
Implementation of Brier Score.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

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

        loss = torch.sum(squared_diff) / float(input.shape[0])
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
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)

        predicted_probs = F.softmax(input, dim=1)
        predicted_probs, pred_labels = torch.max(predicted_probs, 1)
        correct_mask = torch.where(torch.eq(pred_labels, target),
                          torch.ones(pred_labels.shape).to(self.device),
                          torch.zeros(pred_labels.shape).to(self.device))

        with torch.no_grad():
            c_minus_r = (correct_mask - predicted_probs) ** self.gamma
        
        loss = -1 * c_minus_r * logpt

        if self.size_average: return loss.mean()
        else: return loss.sum()
        