import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from calibrator.calibrator import Calibrator
from calibrator.metrics import ECE
import numpy as np

class ConsistencyLoss(nn.Module):
    def __init__(self, size_average=False, gamma=3.0):
        super(ConsistencyLoss, self).__init__()
        self.size_average = size_average
        self.gamma = gamma
    def forward(self, input, target, calibrated_probability):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input, dim=-1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = logpt.exp()

        prob = calibrated_probability.gather(1,target).view(-1)

        with torch.no_grad():
            weight = (1-prob)**self.gamma

        loss = -1 * weight * logpt

        if self.size_average: return loss.mean()
        else: return loss.sum()

