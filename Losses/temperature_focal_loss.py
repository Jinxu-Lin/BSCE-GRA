import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from Metrics.metrics import ECELoss

class TemperatureFocalLoss(nn.Module):
    def __init__(self, gamma=0, size_average=False, cross_validate='ece', temperature=1.0):
        super(TemperatureFocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average
        self.cross_validate = cross_validate
        self.temperature = temperature

    def forward(self, logits, target):
        if logits.dim()>2:
            logits = logits.view(logits.size(0),logits.size(1),-1)  # N,C,H,W => N,C,H*W
            logits = logits.transpose(1,2)    # N,C,H*W => N,H*W,C
            logits = logits.contiguous().view(-1,logits.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)
        

        logpt = F.log_softmax(logits, dim=-1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        
        # Calculate the temperature-scaled probabilities
        pt = F.softmax(self.temperature_scale(logits), dim=-1)
        pt = pt.gather(1, target).view(-1)
        with torch.no_grad():
            weight = (1-pt)**self.gamma

        loss = -1 * weight * logpt

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