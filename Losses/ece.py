import torch
import torch.nn as nn
import torch.nn.functional as F

class ECELoss(nn.Module):
    def __init__(self, size_average=False, n_bins=5):
        super(ECELoss, self).__init__()
        self.size_average = size_average
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        softmaxes = F.softmax(input, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(target)
        with torch.no_grad():
            ece = torch.zeros(1, device=input.device)
            for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
                # Calculated |confidence - accuracy| in each bin
                in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
                prop_in_bin = in_bin.float().mean()
                if prop_in_bin.item() > 0:
                    accuracy_in_bin = accuracies[in_bin].float().mean()
                    avg_confidence_in_bin = confidences[in_bin].mean()
                    ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        logpt = F.log_softmax(input, dim=-1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)

        loss = -1 * ece * logpt

        if self.size_average: return loss.mean()
        else: return loss.sum()