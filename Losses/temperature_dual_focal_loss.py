import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from Metrics.metrics import ECELoss

ps = [0.2, 0.5]
gammas = [5.0, 3.0]
i = 0
gamma_dic = {}
for p in ps:
    gamma_dic[p] = gammas[i]
    i += 1

class TemperatureDualFocalLoss(nn.Module):
    def __init__(self, gamma=0, size_average=False, cross_validate='ece', temperature=1.0):
        super(TemperatureDualFocalLoss, self).__init__()
        self.gamma = gamma
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
        logpt = logpt.gather(1,labels)
        logpt = logpt.view(-1)
        
        # Calculate the temperature-scaled probabilities
        scaled_probability = F.softmax(self.temperature_scale(logits), dim=-1)
        pt = scaled_probability.gather(1, labels).view(-1)

        p_j_mask = torch.lt(scaled_probability, pt.reshape(pt.shape[0], 1)) * 1  # mask all logit larger and equal than p_k
        p_j = torch.topk(p_j_mask * scaled_probability, 1)[0].squeeze()
        
        weight = (1-pt+p_j)**self.gamma

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
    
    
class TemperatureDualFocalLossGra(nn.Module):
    def __init__(self, gamma=0, size_average=False, cross_validate='ece', temperature=1.0):
        super(TemperatureDualFocalLossGra, self).__init__()
        self.gamma = gamma
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
        logpt = logpt.gather(1,labels)
        logpt = logpt.view(-1)
        
        # Calculate the temperature-scaled probabilities
        scaled_probability = F.softmax(self.temperature_scale(logits), dim=-1)
        pt = scaled_probability.gather(1, labels).view(-1)

        p_j_mask = torch.lt(scaled_probability, pt.reshape(pt.shape[0], 1)) * 1  # mask all logit larger and equal than p_k
        p_j = torch.topk(p_j_mask * scaled_probability, 1)[0].squeeze()
        
        with torch.no_grad():
            weight = (1-pt+p_j)**self.gamma

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


class TemperatureDualFocalLossAdaptive(nn.Module):
    def __init__(self, gamma=0, size_average=False, cross_validate='ece', temperature=1.0):
        super(TemperatureDualFocalLossAdaptive, self).__init__()
        self.gamma = gamma
        self.size_average = size_average
        self.cross_validate = cross_validate
        self.temperature = temperature

    def forward(self, logits, labels):
        if logits.dim()>2:
            logits = logits.view(logits.size(0),logits.size(1),-1)  # N,C,H,W => N,C,H*W
            logits = logits.transpose(1,2)    # N,C,H*W => N,H*W,C
            logits = logits.contiguous().view(-1,logits.size(2))   # N,H*W,C => N*H*W,C
        labels = labels.view(-1,1)

        nll = F.log_softmax(logits, dim=-1).gather(1,labels).view(-1)
        
        # Calculate the temperature-scaled probabilities
        scaled_probability = F.softmax(self.temperature_scale(logits), dim=-1)
        confidence = scaled_probability.gather(1, labels).view(-1)

        p_j_mask = torch.lt(scaled_probability, confidence.reshape(confidence.shape[0], 1)) * 1  # mask all logit larger and equal than p_k
        p_j = torch.topk(p_j_mask * scaled_probability, 1)[0].squeeze()
        
        weight = (1-confidence+p_j)**self.gamma

        loss = -1 * weight * nll

        if self.size_average: return loss.mean()
        else: return loss.sum()

    def temperature_scale(self, logits):
        scaled_logits = logits / self.temperature
        return scaled_logits
    
    def get_gamma_list(self, confidence):
        gamma_list = []
        batch_size = confidence.shape[0]
        for i in range(batch_size):
            confidence_sample = confidence[i].item()
            if (confidence_sample >= 0.5):
                gamma_list.append(self.gamma)
                continue
            # Choosing the gamma for the sample
            for key in sorted(gamma_dic.keys()):
                if confidence_sample < key:
                    gamma_list.append(gamma_dic[key])
                    break
        return torch.tensor(gamma_list).to(confidence.device)

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