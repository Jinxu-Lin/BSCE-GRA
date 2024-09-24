import os
import sys
import torch
import random
import argparse
from torch import nn
import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn
import numpy as np

# Import dataloaders
import Data.cifar10 as cifar10
import Data.cifar100 as cifar100
import Data.tiny_imagenet as tiny_imagenet

# Import network architectures
from Net.resnet_tiny_imagenet import resnet50 as resnet50_ti
from Net.resnet import resnet50, resnet110
from Net.wide_resnet import wide_resnet_cifar
from Net.densenet import densenet121

# Import metrics to compute
from Metrics.metrics import test_classification_net_logits
from Metrics.metrics import ECELoss, AdaptiveECELoss, ClasswiseECELoss, BrierScoreLoss, ConfAccLoss

# Import temperature scaling and NLL utilities
from Utils.temperature_scaling import ModelWithTemperature


# Dataset params
dataset_num_classes = {
    'cifar10': 10,
    'cifar100': 100,
    'tiny_imagenet': 200
}

dataset_loader = {
    'cifar10': cifar10,
    'cifar100': cifar100,
    'tiny_imagenet': tiny_imagenet
}

# Mapping model name to model function
models = {
    'resnet50': resnet50,
    'resnet50_ti': resnet50_ti,
    'resnet110': resnet110,
    'wide_resnet': wide_resnet_cifar,
    'densenet121': densenet121
}


def set_seeds(seed):
    
    random.seed(seed)
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parseArgs():
    default_dataset = 'cifar10'
    dataset_root = './'
    model = 'resnet50'
    save_loc = './model/best/'
    saved_model_name = 'resnet50_cross_entropy_350.model'
    num_bins = 15
    model_name = None
    train_batch_size = 128
    test_batch_size = 128
    cross_validation_error = 'ece'

    gamma = 1.0
    gamma2 = 1.0
    gamma3 = 1.0
    bsce_norm = 1

    parser = argparse.ArgumentParser(
        description="Evaluating a single model on calibration metrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--seed", type=int, default=default_dataset,
                        dest="seed", help='dataset to test on')
    parser.add_argument("--dataset", type=str, default=default_dataset,
                        dest="dataset", help='dataset to test on')
    parser.add_argument("--dataset-root", type=str, default=dataset_root,
                        dest="dataset_root", help='root path of the dataset (for tiny imagenet)')
    parser.add_argument("--model-name", type=str, default=model_name,
                        dest="model_name", help='name of the model')
    parser.add_argument("--model", type=str, default=model, dest="model",
                        help='Model to test')
    parser.add_argument("--save-path", type=str, default=save_loc,
                        dest="save_loc",
                        help='Path to import the model')
    parser.add_argument("--saved_model_name", type=str, default=saved_model_name,
                        dest="saved_model_name", help="file name of the pre-trained model")
    parser.add_argument("--num-bins", type=int, default=num_bins, dest="num_bins",
                        help='Number of bins')
    parser.add_argument("-g", action="store_true", dest="gpu",
                        help="Use GPU")
    parser.set_defaults(gpu=True)
    parser.add_argument("-da", action="store_true", dest="data_aug",
                        help="Using data augmentation")
    parser.set_defaults(data_aug=True)
    parser.add_argument("-b", type=int, default=train_batch_size,
                        dest="train_batch_size", help="Batch size")
    parser.add_argument("-tb", type=int, default=test_batch_size,
                        dest="test_batch_size", help="Test Batch size")
    parser.add_argument("--cverror", type=str, default=cross_validation_error,
                        dest="cross_validation_error", help='Error function to do temp scaling')
    parser.add_argument("-log", action="store_true", dest="log",
                        help="whether to print log data")
    parser.add_argument("--loss", type=str, default='dual_focal_loss')

    parser.add_argument("--lamda", type=float, default=1.0)
    parser.add_argument("--gamma-schedule", type=int, default=0,
                        dest="gamma_schedule", help="Schedule gamma or not")
    parser.add_argument("--gamma", type=float, default=gamma,
                        dest="gamma", help="Gamma for focal components")
    parser.add_argument("--gamma2", type=float, default=gamma2,
                        dest="gamma2", help="Gamma for different focal components")
    parser.add_argument("--gamma3", type=float, default=gamma3,
                        dest="gamma3", help="Gamma for different focal components")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--bsce-norm", type=int, default=bsce_norm, 
                        dest="bsce_norm", help="Normalization for bsce")
    parser.add_argument("--epoch", type=int, default=350)


    return parser.parse_args()


def get_logits_labels(data_loader, net):
    logits_list = []
    labels_list = []
    net.eval()
    with torch.no_grad():
        for data, label in data_loader:
            data = data.cuda()
            logits = net(data)
            logits_list.append(logits)
            labels_list.append(label)
        logits = torch.cat(logits_list).cuda()
        labels = torch.cat(labels_list).cuda()
    return logits, labels


def loss_function_save_name(loss_function,
                            scheduled=False,
                            temperature=1.0,
                            gamma=1.0,
                            gamma1=1.0,
                            gamma2=1.0,
                            gamma3=1.0,
                            lamda=1.0,
                            bsce_norm=1,
                            num_bins=5):
    res_dict = {
        'cross_entropy': 'cross_entropy',
        'cross_entropy_exp': 'cross_entropy_exp_temperature_' + str(temperature),
        'cross_entropy_weight_bs': 'cross_entropy_weight_bs_temperature_' + str(temperature),
        'focal_loss': 'focal_loss_gamma_' + str(gamma),
        'focal_loss_gra': 'focal_loss_gra_gamma_' + str(gamma),
        'focal_loss_exp': 'focal_loss_exp_gamma_' + str(gamma) + '_temperature_' + str(temperature),
        'focal_loss_adaptive': 'focal_loss_adaptive_gamma_' + str(gamma),
        'focal_loss_adaptive_gra': 'focal_loss_adaptive_gra_gamma_' + str(gamma),
        'dual_focal_loss': 'dual_focal_loss_gamma_' + str(gamma),
        'dual_focal_loss_gra': 'dual_focal_loss_gra_gamma_' + str(gamma),
        'ada_focal': 'ada_focal',
        'mmce': 'mmce_lamda_' + str(lamda),
        'mmce_gra': 'mmce_gra',
        'mmce_weighted': 'mmce_weighted_lamda_' + str(lamda),
        'brier_score': 'brier_score',
        'brier_score_exp': 'brier_score_exp_temperature_' + str(temperature),
        'brier_score_exp_no_clipping': 'brier_score_exp_no_clipping_temperature_' + str(temperature),
        'brier_score_exp_no_minus': 'brier_score_exp_no_minus_temperature_' + str(temperature),
        'brier_score_exp_pure': 'brier_score_exp_pure',
        'bsce': 'bsce_gamma_' + str(gamma) + '_norm_' + str(bsce_norm),
        'bsce_gra': 'bsce_gra_gamma_' + str(gamma) + '_norm_' + str(bsce_norm),
        'bsce_adaptive_gra': 'bsce_adaptive_gra_gamma_' + str(gamma) + '_norm_' + str(bsce_norm),
        'ece_loss': 'ece_loss_' + str(num_bins),
        'tlbs': 'tlbs_gamma_' + str(gamma),
    }
    res_str = res_dict[loss_function]
    return res_str


if __name__ == "__main__":

    # Checking if GPU is available
    cuda = False
    if (torch.cuda.is_available()):
        cuda = True

    # Setting additional parameters
    args = parseArgs()
    set_seeds(args.seed)
    device = torch.device("cuda" if cuda else "cpu")

    if args.model_name is None:
        args.model_name = args.model

    dataset = args.dataset
    dataset_root = args.dataset_root
    model_name = args.model_name
    save_loc = '/home/jinxulin/UQ/model/' + args.dataset + '-' + args.model + '-' + args.loss + '/' + str(args.seed) + '/epoch/'
    saved_model_name = args.model + '_' + loss_function_save_name(args.loss, args.gamma_schedule, args.temperature, args.gamma, args.gamma, args.gamma2, args.gamma3, args.lamda, args.bsce_norm, args.num_bins) + \
          "_" + str(args.epoch) + ".model"
    num_bins = args.num_bins
    cross_validation_error = args.cross_validation_error

    # Taking input for the dataset
    num_classes = dataset_num_classes[dataset]
    if (args.dataset == 'tiny_imagenet'):
        val_loader = dataset_loader[args.dataset].get_data_loader(
            root=args.dataset_root,
            split='val',
            batch_size=args.test_batch_size,
            pin_memory=args.gpu)

        test_loader = dataset_loader[args.dataset].get_data_loader(
            root=args.dataset_root,
            split='val',
            batch_size=args.test_batch_size,
            pin_memory=args.gpu)
    else:
        _, val_loader = dataset_loader[args.dataset].get_train_valid_loader(
            batch_size=args.train_batch_size,
            augment=args.data_aug,
            random_seed=1,
            pin_memory=args.gpu
        )

        test_loader = dataset_loader[args.dataset].get_test_loader(
            batch_size=args.test_batch_size,
            pin_memory=args.gpu
        )

    model = models[model_name]

    net = model(num_classes=num_classes, temp=1.0)
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

    model_path = os.path.join(save_loc, saved_model_name)
    net.load_state_dict(torch.load(model_path))

    # conf_acc = ConfAccLoss().cuda()
    nll_criterion = nn.CrossEntropyLoss().cuda()
    ece_criterion = ECELoss().cuda()
    adaece_criterion = AdaptiveECELoss().cuda()
    cece_criterion = ClasswiseECELoss().cuda()
    bs_criterion = BrierScoreLoss().cuda()

    logits, labels = get_logits_labels(test_loader, net)
    conf_matrix, p_accuracy, _, _, _ = test_classification_net_logits(logits, labels)

    # bin_stats = conf_acc(logits, labels)
    # bin_stats = bin_stats.cpu().numpy()
    # np.save(save_loc+str(args.epoch)+'_bin_stats.npy', bin_stats)

    p_ece = ece_criterion(logits, labels).item()
    p_adaece = adaece_criterion(logits, labels).item()
    p_cece = cece_criterion(logits, labels).item()
    p_nll = nll_criterion(logits, labels).item()
    p_bs = bs_criterion(logits,labels).item()

    res_str = '{:s}&{:.4f}&{:.4f}&{:.4f}&{:.4f}&{:.4f}&{:.4f}'.format(saved_model_name,  1-p_accuracy,  p_nll,  p_ece,  p_adaece, p_cece, p_bs)

    # Printing the required evaluation metrics
    if args.log:
        print(conf_matrix)
        print('Test error: {:.2f}%'.format((1 - p_accuracy) * 100))
        # print('Test NLL: {:.2f}'.format(p_nll * 100))
        print('ECE: {:.2f}'.format(p_ece * 100))
        print('AdaECE: {:.2f}'.format(p_adaece * 100))
        print('Classwise ECE: {:.2f}'.format(p_cece * 100))
        print('Brier Score: {:.2f}'.format(p_bs * 100))



    scaled_model = ModelWithTemperature(net, args.log)
    scaled_model.set_temperature(test_loader, cross_validate=cross_validation_error)
    T_opt = scaled_model.get_temperature()
    logits, labels = get_logits_labels(test_loader, scaled_model)
    conf_matrix, accuracy, _, _, _ = test_classification_net_logits(logits, labels)

    ece = ece_criterion(logits, labels).item()
    adaece = adaece_criterion(logits, labels).item()
    cece = cece_criterion(logits, labels).item()
    nll = nll_criterion(logits, labels).item()
    bs = bs_criterion(logits,labels).item()

    res_str += '&{:.4f}({:.2f})&{:.4f}&{:.4f}&{:.4f}&{:.4f}'.format(nll,  T_opt,  ece,  adaece, cece, bs)

    if args.log:
        print ('Optimal temperature: ' + str(T_opt))
        print (conf_matrix)
        print('Test error: {:.2f}%'.format((1 - accuracy) * 100))
        print('Test NLL: {:.2f}'.format(nll * 100))
        print('ECE: {:.2f}'.format(ece * 100))
        print('AdaECE: {:.2f}'.format(adaece * 100))
        print('Classwise ECE: {:.2f}'.format(cece * 100))
        print('Brier Score: {:.2f}'.format(bs * 100))

    # # Test NLL & ECE & AdaECE & Classwise ECE
    # print(res_str)