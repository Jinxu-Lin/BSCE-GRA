'''
Script for training models.
'''

from torch import optim
import torch
import torch.utils.data
import argparse
import torch.backends.cudnn as cudnn
import torchvision
import random
import json
import sys
import time
import os
import wandb
import csv

from calibrator import LocalCalibrator

# Import dataloaders
import Data.cifar10 as cifar10
import Data.cifar100 as cifar100
import Data.tiny_imagenet as tiny_imagenet

# Import network models
from Net.resnet import resnet50, resnet110
from Net.resnet_tiny_imagenet import resnet50 as resnet50_ti
from Net.wide_resnet import wide_resnet_cifar
from Net.densenet import densenet121
from Net.vit import vit
# Import loss functions
from Losses.loss import set_loss_function
# Import train and validation utilities
from Utils.train_utils import train_single_epoch, train_single_epoch_warmup
from Utils.eval_utils import evaluate_dataset, evaluate_dataset_train

# Import validation metrics
from Metrics.metrics import test_classification_net
from Metrics.metrics import ECELoss


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


models = {
    'resnet50': resnet50,
    'resnet50_ti': resnet50_ti,
    'resnet110': resnet110,
    'wide_resnet': wide_resnet_cifar,
    'densenet121': densenet121,
    'vit': vit
}

temperature_loss_list = [
    'temperature_focal_loss',
    'temperature_focal_loss_adaptive',
    'temperature_focal_loss_gra',
    'temperature_focal_loss_adaptive_gra',
    'temperature_dual_focal_loss',
    'temperature_dual_focal_loss_gra',
    'temperature_bsce',
    'temperature_bsce_gra',
]

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
        'dual_focal_loss_exp': 'dual_focal_loss_exp_gamma_' + str(gamma) + '_temperature_' + str(temperature),
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
        'tlbs': 'tlbs_gamma_' + str(gamma),
        'consistency': 'consistency',
        'ece_loss': 'ece_loss_' + str(num_bins),
        'temperature_focal_loss': 'temperature_focal_loss_gamma_' + str(gamma),
        'temperature_focal_loss_adaptive': 'temperature_focal_loss_adaptive_gamma_' + str(gamma),
        'temperature_focal_loss_gra': 'temperature_focal_loss_gra_gamma_' + str(gamma),
        'temperature_focal_loss_adaptive_gra': 'temperature_focal_loss_adaptive_gra_gamma_' + str(gamma),
        'temperature_dual_focal_loss': 'temperature_dual_focal_loss_gamma_' + str(gamma),
        'temperature_dual_focal_loss_gra': 'temperature_dual_focal_loss_gra_gamma_' + str(gamma),
        'temperature_bsce': 'temperature_bsce_gamma_' + str(gamma) + '_norm_' + str(bsce_norm),
        'temperature_bsce_gra': 'temperature_bsce_gra_gamma_' + str(gamma) + '_norm_' + str(bsce_norm),
    }
    if (loss_function == 'focal_loss' and scheduled == True):
        res_str = 'focal_loss_scheduled_gamma_' + str(gamma1) + '_' + str(gamma2) + '_' + str(gamma3)
    else:
        res_str = res_dict[loss_function]
    return res_str


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
    train_batch_size = 128
    test_batch_size = 128
    learning_rate = 0.1
    momentum = 0.9
    optimiser = "sgd"
    loss = "cross_entropy"
    warm_up_epochs = 0

    gamma = 1.0
    gamma2 = 1.0
    gamma3 = 1.0
    lamda = 1.0
    gamma_schedule_step1 = 100
    gamma_schedule_step2 = 250
    temperature = 1.0
    bsce_norm = 1
    num_bins = 15
    adafocal_lambda = 1.0
    adafocal_gamma_initial = 1.0
    adafocal_gamma_max = 20.0
    adafocal_gamma_min = -2.0
    adafocal_switch_pt = 0.2
    update_gamma_every = -1

    weight_decay = 5e-4
    log_interval = 50
    save_interval = 50
    save_loc = './model/'
    model_name = None
    saved_model_name = "resnet50_cross_entropy_350.model"
    load_loc = './model/'
    model = "resnet50"
    epoch = 350
    first_milestone = 150 #Milestone for change in lr
    second_milestone = 250 #Milestone for change in lr

    parser = argparse.ArgumentParser(
        description="Training for calibration.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # data loader
    parser.add_argument("--dataset", type=str, default=default_dataset,
                        dest="dataset", help='dataset to train on')
    parser.add_argument("--dataset-root", type=str, default=dataset_root,
                        dest="dataset_root", help='root path of the dataset (for tiny imagenet)')
    parser.add_argument("--data-aug", action="store_true", dest="data_aug")
    parser.set_defaults(data_aug=True)

    # train
    parser.add_argument("--seed", type=int, default=1,
                        dest="seed", help='random seed')
    parser.add_argument("-g", action="store_true", dest="gpu",
                        help="Use GPU")
    parser.set_defaults(gpu=True)
    parser.add_argument("--load", action="store_true", dest="load",
                        help="Load from pretrained model")
    parser.set_defaults(load=False)
    parser.add_argument("-b", type=int, default=train_batch_size,
                        dest="train_batch_size", help="Batch size")
    parser.add_argument("-tb", type=int, default=test_batch_size,
                        dest="test_batch_size", help="Test Batch size")
    parser.add_argument("-e", type=int, default=epoch, dest="epoch",
                        help='Number of training epochs')
    parser.add_argument("--lr", type=float, default=learning_rate,
                        dest="learning_rate", help='Learning rate')
    parser.add_argument("--mom", type=float, default=momentum,
                        dest="momentum", help='Momentum')
    parser.add_argument("--nesterov", action="store_true", dest="nesterov",
                        help="Whether to use nesterov momentum in SGD")
    parser.set_defaults(nesterov=False)
    parser.add_argument("--decay", type=float, default=weight_decay,
                        dest="weight_decay", help="Weight Decay")
    parser.add_argument("--opt", type=str, default=optimiser,
                        dest="optimiser",
                        help='Choice of optimisation algorithm')
    parser.add_argument("--warm-up-epochs", type=int, default=warm_up_epochs, dest="warm_up_epochs",
                        help="Warm up epochs")

    # loss
    parser.add_argument("--loss", type=str, default=loss, dest="loss_function",
                        help="Loss function to be used for training")
    parser.add_argument("--loss-mean", action="store_true", dest="loss_mean",
                        help="whether to take mean of loss instead of sum to train")
    parser.set_defaults(loss_mean=False)
    parser.add_argument("--temperature", type=float, default=temperature,
                        dest="temperature", help="Temperature for cross entropy")
    parser.add_argument("--gamma", type=float, default=gamma,
                        dest="gamma", help="Gamma for focal components")
    parser.add_argument("--gamma2", type=float, default=gamma2,
                        dest="gamma2", help="Gamma for different focal components")
    parser.add_argument("--gamma3", type=float, default=gamma3,
                        dest="gamma3", help="Gamma for different focal components")
    parser.add_argument("--lamda", type=float, default=lamda,
                        dest="lamda", help="Regularization factor")
    parser.add_argument("--num-bins", type=int, default=num_bins,
                        dest="num_bins", help="The amount of bins for ece")
    parser.add_argument("--gamma-schedule", type=int, default=0,
                        dest="gamma_schedule", help="Schedule gamma or not")
    parser.add_argument("--gamma-schedule-step1", type=int, default=gamma_schedule_step1,
                        dest="gamma_schedule_step1", help="1st step for gamma schedule")
    parser.add_argument("--gamma-schedule-step2", type=int, default=gamma_schedule_step2,
                        dest="gamma_schedule_step2", help="2nd step for gamma schedule")
    parser.add_argument("--bsce-norm", type=int, default=bsce_norm, 
                        dest="bsce_norm", help="Normalization for bsce")
    parser.add_argument("--adafocal-lambda", type=float, default=adafocal_lambda, dest="adafocal_lambda", help="lambda for adafocal.")
    parser.add_argument("--adafocal-gamma-initial", type=float, default=adafocal_gamma_initial, dest="adafocal_gamma_initial", help="Initial gamma for each bin.")
    parser.add_argument("--adafocal-gamma-max", type=float, default=adafocal_gamma_max, dest="adafocal_gamma_max", help="Maximum cutoff value for gamma.")
    parser.add_argument("--adafocal-gamma-min", type=float, default=adafocal_gamma_min, dest="adafocal_gamma_min", help="Minimum cutoff value for gamma.")
    parser.add_argument("--adafocal-switch-pt", type=float, default=adafocal_switch_pt, dest="adafocal_switch_pt", help="Gamma at which to switch to inverse-focal loss.")
    parser.add_argument("--update-gamma-every", type=int, default=update_gamma_every, dest="update_gamma_every", help="Update gamma every nth batch. If -1, update after epoch end.")
    parser.add_argument("--size-average", action="store_true", dest="size_average",
                        help="Whether to take mean of loss instead of sum")
    
    
    # log and save
    parser.add_argument("--log-interval", type=int, default=log_interval,
                        dest="log_interval", help="Log Interval on Terminal")
    parser.add_argument("--save-interval", type=int, default=save_interval,
                        dest="save_interval", help="Save Interval on Terminal")
    parser.add_argument("--saved-model-name", type=str, default=saved_model_name,
                        dest="saved_model_name", help="file name of the pre-trained model")
    parser.add_argument("--save-path", type=str, default=save_loc,
                        dest="save_loc",
                        help='Path to export the model')
    parser.add_argument("--model-name", type=str, default=model_name,
                        dest="model_name",
                        help='name of the model')
    parser.add_argument("--load-path", type=str, default=load_loc,
                        dest="load_loc",
                        help='Path to load the model from')
    parser.add_argument("--wandb-offline", action="store_true", dest="wandb_offline",
                        help="Run wandb in offline mode")

    parser.add_argument("--model", type=str, default=model, dest="model",
                        help='Model to train')
    parser.add_argument("--first-milestone", type=int, default=first_milestone,
                        dest="first_milestone", help="First milestone to change lr")
    parser.add_argument("--second-milestone", type=int, default=second_milestone,
                        dest="second_milestone", help="Second milestone to change lr")

    return parser.parse_args()


if __name__ == "__main__":

    args = parseArgs()
    set_seeds(args.seed)

    # Setting model name
    if args.model_name is None:
        args.model_name = args.model
    save_path = os.path.join(args.save_loc, args.dataset, args.model_name)
    model_name = args.dataset+'-'+args.model_name+'-'+args.loss_function
    save_loc = os.path.join(save_path, model_name, str(args.seed))

    os.makedirs(os.path.join(save_loc, 'csv'), exist_ok=True)
    csv_file_path = os.path.join(save_loc, 'csv')
    csv_file = args.model_name + '_' + \
            loss_function_save_name(args.loss_function, args.gamma_schedule, args.temperature, args.gamma, args.gamma, args.gamma2, args.gamma3, args.lamda, args.bsce_norm, args.num_bins) + \
            '.csv'
    mode = 'a' if args.load else 'w'
    with open(csv_file_path+'/'+csv_file, mode=mode, newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["args"] + [f"{key}={value}" for key, value in vars(args).items()])
        writer.writerow(["epoch", "train_loss", "val_loss", "val_acc", "val_ece", "test_acc", "test_ece"])

    cuda = False
    if (torch.cuda.is_available() and args.gpu):
        cuda = True
    device = torch.device("cuda" if cuda else "cpu")
    print("CUDA set: " + str(cuda))

    num_classes = dataset_num_classes[args.dataset]

    # Choosing the model to train
    net = models[args.model](num_classes=num_classes)

    if args.gpu is True:
        net.cuda()
        net = torch.nn.DataParallel(
            net, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True

    start_epoch = 0
    num_epochs = args.epoch
    if args.load:
        load_path = save_loc + '/epoch/' + args.saved_model_name
        net.load_state_dict(torch.load(load_path))
        start_epoch = int(args.saved_model_name[args.saved_model_name.rfind('_')+1:args.saved_model_name.rfind('.model')])

    if args.optimiser == "sgd":
        opt_params = net.parameters()
        optimizer = optim.SGD(opt_params,
                              lr=args.learning_rate,
                              momentum=args.momentum,
                              weight_decay=args.weight_decay,
                              nesterov=args.nesterov)
    elif args.optimiser == "adam":
        opt_params = net.parameters()
        optimizer = optim.Adam(opt_params,
                               lr=args.learning_rate,
                               weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[args.first_milestone, args.second_milestone], gamma=0.1)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch, eta_min=0)

    if (args.dataset == 'tiny_imagenet'):
        train_loader = dataset_loader[args.dataset].get_data_loader(
            root=args.dataset_root,
            split='train',
            batch_size=args.train_batch_size,
            pin_memory=args.gpu)

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
        train_loader, val_loader = dataset_loader[args.dataset].get_train_valid_loader(
            batch_size=args.train_batch_size,
            augment=args.data_aug,
            random_seed=1,
            pin_memory=args.gpu
        )

        test_loader = dataset_loader[args.dataset].get_test_loader(
            batch_size=args.test_batch_size,
            pin_memory=args.gpu
        )
        
    # Set loss function
    loss_function = set_loss_function(args, device)
    # Set Calibrator
    if args.loss_function == 'consistency':
        calibrator = LocalCalibrator()
    else:
        calibrator = None
    # Set temperature
    temperature = 1.0

    # load from checkpoint
    if args.load:
        for epoch in range(0, start_epoch):
            scheduler.step()

    # warm up
    eps_opt = 1
    if start_epoch < args.warm_up_epochs:

        for epoch in range(start_epoch, args.warm_up_epochs):

            # Gamma schedule for focal loss
            if (args.loss_function == 'focal_loss' and args.gamma_schedule == 1):
                if (epoch < args.gamma_schedule_step1):
                    gamma = args.gamma
                elif (epoch >= args.gamma_schedule_step1 and epoch < args.gamma_schedule_step2):
                    gamma = args.gamma2
                else:
                    gamma = args.gamma3
            else:
                gamma = args.gamma
            
            train_loss, _, labels_list, fulldataset_logits, predictions_list, confidence_list = train_single_epoch_warmup(args,
                                            epoch,
                                            net,
                                            train_loader,
                                            val_loader,
                                            optimizer,
                                            device,
                                            loss_function=loss_function,
                                            num_labels=num_classes,
                                            )
            scheduler.step()

            # if args.loss_function == 'ece_loss':
            #     train_ece, train_bin_dict, train_adaece, train_adabin_dict, train_classwise_ece, train_classwise_dict = evaluate_dataset_train(labels_list, fulldataset_logits, predictions_list, confidence_list, num_bins=args.num_bins)
            #     loss_function.update_bin_stats(train_bin_dict, train_adabin_dict, train_classwise_dict)

            (val_loss, val_confusion_matrix, val_acc, val_ece, val_bin_dict,
            val_adaece, val_adabin_dict, val_mce, val_classwise_ece, val_classwise_dict, val_logits, val_labels) = evaluate_dataset(net, val_loader, device, num_bins=args.num_bins, num_labels=num_classes)

            if args.loss_function == 'ece_loss':
                loss_function.update_bin_stats(val_bin_dict, val_adabin_dict, val_classwise_dict)
            elif args.loss_function == 'consistency':
                eps_opt = calibrator.fit(val_logits, torch.tensor(val_labels).to(device))
            elif args.loss_function in temperature_loss_list:
                loss_function.update_temperature(val_logits, torch.tensor(val_labels).to(device))
            
            (test_loss, test_confusion_matrix, test_acc, test_ece, test_bin_dict, 
            test_adaece, test_adabin_dict, test_mce, test_classwise_ece, test_classwise_dict, test_logits, test_labels) = evaluate_dataset(net, test_loader, device, num_bins=args.num_bins, num_labels=num_classes)

            with open(csv_file_path+'/'+csv_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([epoch, train_loss, val_loss, val_acc, val_ece, test_acc, test_ece])

        start_epoch = args.warm_up_epochs

    best_val_acc = 0
    best_ece = 1.0


    for epoch in range(start_epoch, num_epochs):
        
        # Gamma schedule for focal loss
        if(args.loss_function == 'focal_loss' and args.gamma_schedule == 1):
            if (epoch < args.gamma_schedule_step1):
                gamma = args.gamma
            elif (epoch >= args.gamma_schedule_step1 and epoch < args.gamma_schedule_step2):
                gamma = args.gamma2
            else:
                gamma = args.gamma3
        else:
            gamma = args.gamma
    
        train_loss, _, labels_list, fulldataset_logits, predictions_list, confidence_list = train_single_epoch(args,
                                        epoch,
                                        net,
                                        train_loader,
                                        val_loader,
                                        optimizer,
                                        device,
                                        loss_function=loss_function,
                                        num_labels=num_classes,
                                        calibrator=calibrator,
                                    )
        
        scheduler.step()

        if args.loss_function == 'ece_loss':
            train_ece, train_bin_dict, train_adaece, train_adabin_dict, train_classwise_ece, train_classwise_dict = evaluate_dataset_train(labels_list, fulldataset_logits, predictions_list, confidence_list, num_bins=args.num_bins)
            loss_function.update_bin_stats(train_bin_dict, train_adabin_dict, train_classwise_dict)

        # This evaluates the current model on the validation set to collect various performance statistics.
        # This calls the "evaluate_dataset" function implemented in utils/eval_utils.py
        (val_loss, val_confusion_matrix, val_acc, val_ece, val_bin_dict,
        val_adaece, val_adabin_dict, val_mce, val_classwise_ece, val_classwise_dict, val_logits, val_labels) = evaluate_dataset(net, val_loader, device, num_bins=args.num_bins, num_labels=num_classes)

        if args.loss_function == 'ece_loss':
            loss_function.update_bin_stats(val_bin_dict, val_adabin_dict, val_classwise_dict)
        elif args.loss_function == 'consistency':
            eps_opt = calibrator.fit(val_logits, torch.tensor(val_labels).to(device))
        elif args.loss_function in temperature_loss_list:
            loss_function.update_temperature(val_logits, torch.tensor(val_labels).to(device))

        (test_loss, test_confusion_matrix, test_acc, test_ece, test_bin_dict, 
        test_adaece, test_adabin_dict, test_mce, test_classwise_ece, test_classwise_dict, test_logits, test_labels) = evaluate_dataset(net, test_loader, device, num_bins=args.num_bins, num_labels=num_classes)

        os.makedirs(os.path.join(save_loc, 'epoch'), exist_ok=True)
            
        if (epoch + 1) % args.save_interval == 0:
            save_name = save_loc + '/epoch/' + \
                        args.model_name + '_' + \
                        loss_function_save_name(args.loss_function, args.gamma_schedule, args.temperature, gamma, args.gamma, args.gamma2, args.gamma3, args.lamda, args.bsce_norm, args.num_bins) + \
                        '_' + str(epoch + 1) + '.model'
            torch.save(net.state_dict(), save_name)

        with open(csv_file_path+'/'+csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch, train_loss, val_loss, val_acc, val_ece, test_acc, test_ece])
    

    results_csv_file = 'results.csv'
    with open(results_csv_file, mode='a', newline='') as results_file:
        results_writer = csv.writer(results_file)
        # 将 test_acc 转换为百分数形式，限制到小数点后两位
        formatted_test_acc = f"{(1-test_acc) * 100:.2f}%"
        
        # 将 ece 相关值限制到小数点后五位并乘以100
        formatted_test_ece = f"{test_ece * 100:.2f}"
        formatted_test_adaece = f"{test_adaece.item() * 100:.2f}"
        formatted_test_classwise_ece = f"{test_classwise_ece.item() * 100:.2f}"

        results_writer.writerow([
            args.dataset, 
            args.model, 
            args.loss_function, 
            args.seed, 
            args.warm_up_epochs, 
            args.gamma, 
            args.bsce_norm, 
            formatted_test_acc, 
            formatted_test_ece, 
            formatted_test_adaece, 
            formatted_test_classwise_ece
        ])