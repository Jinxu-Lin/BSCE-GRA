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
from Losses.focal_loss import FocalLoss, FocalLossGra, FocalLossExp
from Losses.dual_focal_loss import DualFocalLoss, DualFocalLossGra
from Losses.focal_loss_adaptive_gamma import FocalLossAdaptive, FocalLossAdaptiveGra
from Losses.adafocal import AdaFocal
from Losses.mmce import MMCE, MMCEWeighted, MMCEGRA
from Losses.brier_score import BrierScore, BrierScoreExp, BrierScoreExpNoClipping, BrierScoreExpNoMinus, BrierScoreExpPure
from Losses.brier_score import BSCELoss, BSCELossGra, BSCELossAdaptiveGra, TLBSLoss
from Losses.ece import ECELoss
from Losses.dece import DECE
from Losses.ce import CrossEntropy, CrossEntropyExp, CrossEntropyWeightBS

def cross_entropy(args, device):
    return CrossEntropy()

def cross_entropy_exp(args, device):
    return CrossEntropyExp(temperature=args.temperature)

def cross_entropy_weight_bs(args, device):
    return CrossEntropyWeightBS(temperature=args.temperature)

def focal_loss(args, device):
    return FocalLoss(gamma=args.gamma)

def focal_loss_gra(args, device):
    return FocalLossGra(gamma=args.gamma)

def focal_loss_exp(args, device):
    return FocalLossExp(gamma=args.gamma, temperature=args.temperature)

def focal_loss_adaptive(args, device):
    return FocalLossAdaptive(gamma=args.gamma,
                             device=device)

def focal_loss_adaptive_gra(args, device):
    return FocalLossAdaptiveGra(gamma=args.gamma,
                             device=device)

def dual_focal_loss(args, device):
    return DualFocalLoss(gamma=args.gamma)

def dual_focal_loss_gra(args, device):
    return DualFocalLossGra(gamma=args.gamma)

def ada_focal(args, device):
    return AdaFocal(args=args, device=device)

def ece_loss(args, device):
    return ECELoss(n_bins=args.n_bins)

def mmce(args, device):
    return MMCE(device=device, lamda=args.lamda)

def mmce_gra(args, device):
    return MMCEGRA(device=device, lamda=args.lamda)

def mmce_weighted(args, device):
    return MMCEWeighted(device=device, lamda=args.lamda)

def brier_score(args, device):
    return BrierScore()

def brier_score_exp(args, device):
    return BrierScoreExp(temperature=args.temperature)

def brier_score_exp_no_clipping(args, device):
    return BrierScoreExpNoClipping(temperature=args.temperature)

def brier_score_exp_no_minus(args, device):
    return BrierScoreExpNoMinus(temperature=args.temperature)

def brier_score_exp_pure(args, device):
    return BrierScoreExpPure()

def bsce(args, device):
    return BSCELoss(gamma=args.gamma, norm=args.bsce_norm)

def bsce_gra(args, device):
    return BSCELossGra(gamma=args.gamma, norm=args.bsce_norm, size_average=args.size_average)

def bsce_adaptive_gra(args, device):
    return BSCELossAdaptiveGra(gamma=args.gamma, norm=args.bsce_norm, device=device)

def tlbs(args, device):
    return TLBSLoss(gamma=args.gamma, device=device)

def dece(args, device):
    return DECE(device = device, num_bins = args.num_bins, t_a = 100, t_b = 0.01)
    
def set_loss_function(args, device):
    loss_function_dict = {
        'cross_entropy': cross_entropy,
        'cross_entropy_exp': cross_entropy_exp,
        'cross_entropy_weight_bs': cross_entropy_weight_bs,
        'brier_score_exp_no_clipping': brier_score_exp_no_clipping,
        'brier_score_exp_no_minus': brier_score_exp_no_minus,
        'brier_score_exp_pure': brier_score_exp_pure,
        'focal_loss': focal_loss,
        'focal_loss_gra': focal_loss_gra,
        'focal_loss_exp': focal_loss_exp,
        'focal_loss_adaptive': focal_loss_adaptive,
        'focal_loss_adaptive_gra': focal_loss_adaptive_gra,
        'dual_focal_loss': dual_focal_loss,
        'dual_focal_loss_gra': dual_focal_loss_gra,
        'ada_focal': ada_focal,
        'mmce': mmce,
        'mmce_gra': mmce_gra,
        'mmce_weighted': mmce_weighted,
        'brier_score': brier_score,
        'bsce': bsce,
        'brier_score_exp': brier_score_exp,
        'bsce_gra': bsce_gra,
        'bsce_adaptive_gra': bsce_adaptive_gra,
        'ece_loss': ece_loss,
        'tlbs': tlbs,
        'dece': dece,
    }    
    # Get the loss function based on the args.loss_function
    if args.loss_function not in loss_function_dict:
        raise ValueError("Unknown loss function: {}".format(args.loss_function))
    else:
        loss_function = loss_function_dict[args.loss_function](args, device)
        
    return loss_function