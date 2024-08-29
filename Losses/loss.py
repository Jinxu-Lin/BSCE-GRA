'''
Implementation of the following loss functions:
1. Cross Entropy
2. Focal Loss
3. Cross Entropy + MMCE_weighted
4. Cross Entropy + MMCE
5. Brier Score
'''
import torch
from torch.nn import functional as F
from Losses.focal_loss import FocalLoss, FocalLossGra, DualFocalLoss, DualFocalLossGra
from Losses.focal_loss_adaptive_gamma import FocalLossAdaptive, FocalLossAdaptiveGra
from Losses.mmce import MMCE, MMCE_weighted
from Losses.brier_score import BrierScore, BrierScoreExp, BrierScoreExpMinusBrierScore, BSCELoss, BSCELossGra, BSCELossAdaptiveGra, TLBSLoss
from Losses.ece import ECELoss
from Losses.dece import DECE
from Losses.ce import CrossEntropy, CrossEntropyExpMinusCrossEntropy


def cross_entropy(logits, targets, **kwargs):
    return CrossEntropy()(logits, targets, reduction='sum')

def cross_entropy_exp_minus_cross_entropy(logits, targets, **kwargs):
    return CrossEntropyExpMinusCrossEntropy()(logits, targets)

def focal_loss(logits, targets, **kwargs):
    return FocalLoss(gamma=kwargs['gamma'])(logits, targets)

def focal_loss_gra(logits, targets, **kwargs):
    return FocalLossGra(gamma=kwargs['gamma'])(logits, targets)

def focal_loss_adaptive(logits, targets, **kwargs):
    return FocalLossAdaptive(gamma=kwargs['gamma'],
                             device=kwargs['device'])(logits, targets)

def focal_loss_adaptive_gra(logits, targets, **kwargs):
    return FocalLossAdaptiveGra(gamma=kwargs['gamma'],
                             device=kwargs['device'])(logits, targets)

def dual_focal_loss(logits, targets, **kwargs):
    return DualFocalLoss(gamma=kwargs['gamma'])(logits, targets)

def dual_focal_loss_gra(logits, targets, **kwargs):
    return DualFocalLossGra(gamma=kwargs['gamma'])(logits, targets)

def ece_loss(logits, targets, **kwargs):
    return ECELoss(n_bins=kwargs['n_bins'])(logits, targets)

def mmce(logits, targets, **kwargs):
    ce = F.cross_entropy(logits, targets)
    mmce = MMCE(kwargs['device'])(logits, targets)
    return ce + (kwargs['lamda'] * mmce)

def mmce_gra(logits, targets, **kwargs):
    ce = F.cross_entropy(logits, targets)
    with torch.no_grad():
        mmce = MMCE(kwargs['device'])(logits, targets)
    return mmce*ce

def mmce_weighted(logits, targets, **kwargs):
    ce = F.cross_entropy(logits, targets)
    mmce = MMCE_weighted(kwargs['device'])(logits, targets)
    return ce + (kwargs['lamda'] * mmce)

def brier_score(logits, targets, **kwargs):
    return BrierScore()(logits, targets)

def brier_score_exp(logits, targets, **kwargs):
    return BrierScoreExp()(logits, targets)

def brier_score_exp_minus_brier_score(logits, targets, **kwargs):
    return BrierScoreExpMinusBrierScore()(logits, targets)

def bsce(logits, targets, **kwargs):
    return BSCELoss(gamma=kwargs['gamma'], norm=kwargs['bsce_norm'])(logits, targets)

def bsce_gra(logits, targets, **kwargs):
    return BSCELossGra(gamma=kwargs['gamma'], norm=kwargs['bsce_norm'], size_average=kwargs['size_average'])(logits, targets)

def bsce_adaptive_gra(logits, targets, **kwargs):
    return BSCELossAdaptiveGra(gamma=kwargs['gamma'], norm=kwargs['bsce_norm'], device=kwargs['device'])(logits, targets)

def tlbs(logits, targets, **kwargs):
    return TLBSLoss(gamma=kwargs['gamma'], device=kwargs['device'])(logits, targets)

def dece(logits, targets, **kwargs):
    return DECE(device = kwargs["device"], num_bins = kwargs["n_bins"], t_a = 100, t_b = 0.01)(
        logits, targets
    )