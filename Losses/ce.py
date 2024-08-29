import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class CrossEntropy(nn.Module):
    def __init__(self, size_average=False):
        super(CrossEntropy, self).__init__()
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
        pt = logpt.exp()

        loss = -1 * logpt

        if self.size_average: return loss.mean()
        else: return loss.sum()

class CrossEntropyExp(nn.Module):
    def __init__(self, temperature=1.0, size_average=False):
        super(CrossEntropyExp, self).__init__()
        self.size_average = size_average
        self.temperature = temperature
    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = logpt.exp()

        loss = -1 * logpt
        with torch.no_grad():
            weight = torch.exp(
                torch.clamp(loss.detach(), min=0, max=self.temperature) / (self.temperature + 1)
            ) - 1
        loss = loss * weight

        if self.size_average: return loss.mean()
        else: return loss.sum()
