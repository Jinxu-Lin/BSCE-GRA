import torch
import torch.nn as nn
import torch.nn.functional as F
import collections
import numpy as np
class ECELoss(nn.Module):
    def __init__(self, size_average=False):
        super(ECELoss, self).__init__()
        self.size_average = size_average
        self.bin_dict = collections.defaultdict(dict)
        self.bin_ada_dict = collections.defaultdict(dict)
        self.bin_classwise_dict = None
        self.lambda_classwise = 2
    def update_bin_stats(self, bin_dict, bin_ada_dict, bin_classwise_dict):
        self.bin_dict = bin_dict
        self.bin_ada_dict = bin_ada_dict
        self.bin_classwise_dict = bin_classwise_dict

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input, dim=-1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        prob = logpt.exp()

        with torch.no_grad():

            # Calculate ECE
            ece_value = torch.zeros(len(prob)).to(prob.device)
            for i, p in enumerate(prob):
                for bin_index, bin_info in self.bin_dict.items():
                    lower_bound = bin_info.get('lower_bound', 0)
                    upper_bound = bin_info.get('upper_bound', 1)
                    if lower_bound <= p < upper_bound:
                        ece_value[i] = bin_info.get('ece', None)
                        break


            # Calculate AdaECE
            ada_ece_values = torch.zeros(len(prob)).to(prob.device)
            sorted_prob, sorted_indices = torch.sort(prob)
            ada_bin_num = len(self.bin_ada_dict)
            bin_size = len(input) // ada_bin_num

            # Assign calibration gap to each bin
            for i in range(ada_bin_num):
                start_idx = i * bin_size
                end_idx = (i + 1) * bin_size
                ada_ece_values[start_idx:end_idx] = self.bin_ada_dict[i]['calibration_gap']
            
            # Handle the remaining samples
            remaining_samples = len(prob) % ada_bin_num
            if remaining_samples > 0:
                ada_ece_values[-remaining_samples:] = self.bin_ada_dict[ada_bin_num - 1]['calibration_gap']

            ada_ece_values = ada_ece_values[torch.argsort(sorted_indices)]

            # Calculate classwise ECE
            classwise_ece_values = self.bin_classwise_dict[target].view(-1).squeeze()

            weight = ((ece_value).abs()+self.lambda_classwise*(classwise_ece_values).abs())/2


        loss = -1 * weight * logpt

        if self.size_average: return loss.mean()
        else: return loss.sum()