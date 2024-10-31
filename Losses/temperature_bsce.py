import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from Metrics.metrics import ECELoss

from scipy.special import lambertw
import numpy as np

class TemperatureBSCELoss(nn.Module):
    def __init__(self, gamma=0, norm=1, size_average=False, cross_validate='ece', temperature=1.0):
        super(TemperatureBSCELoss, self).__init__()
        self.gamma = gamma
        self.norm = norm
        self.size_average = size_average
        self.cross_validate = cross_validate
        self.temperature = temperature

    def forward(self, logits, labels):
        if logits.dim()>2:
            logits = logits.view(logits.size(0),logits.size(1),-1)  # N,C,H,W => N,C,H*W
            logits = logits.transpose(1,2)    # N,C,H*W => N,H*W,C
            logits = logits.contiguous().view(-1,logits.size(2))   # N,H*W,C => N*H*W,C
        labels = labels.view(-1,1)

        logpt = F.log_softmax(logits, dim=-1)
        nll = logpt.gather(1,labels)
        nll = nll.view(-1)

        # Calculate the temperature-scaled probabilities
        scaled_probability = F.softmax(self.temperature_scale(logits), dim=-1)

        target = torch.FloatTensor(logits.shape).to(labels.get_device())
        target.zero_()
        target.scatter_(1, labels, 1)
        weight = torch.norm(target - scaled_probability, p=self.norm, dim=1) ** self.gamma

        loss = -1 * weight * nll

        if self.size_average: return loss.mean()
        else: return loss.sum()
    
    def temperature_scale(self, logits):
        return logits / self.temperature

    def update_temperature(self, logits, labels):
        
        nll_criterion = nn.CrossEntropyLoss().cuda()
        ece_criterion = ECELoss().cuda()

        nll_val = 10 ** 7
        ece_val = 10 ** 7
        T_opt_nll = 1.0
        T_opt_ece = 1.0
        T = 0.01
        for i in range(1000):
            self.temperature = T
            
            if self.cross_validate == 'ece':
                after_temperature_ece = ece_criterion(self.temperature_scale(logits), labels).item()
                if ece_val > after_temperature_ece:
                    T_opt_ece = T
                    ece_val = after_temperature_ece
            else:
                after_temperature_nll = nll_criterion(self.temperature_scale(logits), labels).item()
                if nll_val > after_temperature_nll:
                    T_opt_nll = T
                    nll_val = after_temperature_nll
            T += 0.01

        if self.cross_validate == 'ece':
            self.temperature = T_opt_ece
        else:
            self.temperature = T_opt_nll

        return self


class TemperatureBSCELossGra(nn.Module):
    def __init__(self, gamma=0, norm=1, size_average=False, cross_validate='ece', temperature=1.0):
        super(TemperatureBSCELossGra, self).__init__()
        self.gamma = gamma
        self.norm = norm
        self.size_average = size_average
        self.cross_validate = cross_validate
        self.temperature = temperature

    def forward(self, logits, labels):
        if logits.dim()>2:
            logits = logits.view(logits.size(0),logits.size(1),-1)  # N,C,H,W => N,C,H*W
            logits = logits.transpose(1,2)    # N,C,H*W => N,H*W,C
            logits = logits.contiguous().view(-1,logits.size(2))   # N,H*W,C => N*H*W,C
        labels = labels.view(-1,1)

        logpt = F.log_softmax(logits, dim=-1)
        nll = logpt.gather(1,labels)
        nll = nll.view(-1)

        # Calculate the temperature-scaled probabilities
        with torch.no_grad():
            scaled_probability = F.softmax(self.temperature_scale(logits), dim=-1)

            target = torch.FloatTensor(logits.shape).to(labels.get_device())
            target.zero_()
            target.scatter_(1, labels, 1)
            weight = torch.norm(target - scaled_probability, p=self.norm, dim=1) ** self.gamma

        loss = -1 * weight * nll

        if self.size_average: return loss.mean()
        else: return loss.sum()

    def temperature_scale(self, logits):
        return logits / self.temperature

    def update_temperature(self, logits, labels):
        
        nll_criterion = nn.CrossEntropyLoss().cuda()
        ece_criterion = ECELoss().cuda()

        nll_val = 10 ** 7
        ece_val = 10 ** 7
        T_opt_nll = 1.0
        T_opt_ece = 1.0
        T = 0.01
        for i in range(1000):
            self.temperature = T
            
            if self.cross_validate == 'ece':
                after_temperature_ece = ece_criterion(self.temperature_scale(logits), labels).item()
                if ece_val > after_temperature_ece:
                    T_opt_ece = T
                    ece_val = after_temperature_ece
            else:
                after_temperature_nll = nll_criterion(self.temperature_scale(logits), labels).item()
                if nll_val > after_temperature_nll:
                    T_opt_nll = T
                    nll_val = after_temperature_nll
            T += 0.01

        if self.cross_validate == 'ece':
            self.temperature = T_opt_ece
        else:
            self.temperature = T_opt_nll

        return self