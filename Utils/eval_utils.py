import torch

from Metrics.metrics import test_classification_net, expected_calibration_error, ClasswiseECELoss, maximum_calibration_error
from Metrics.metrics import adaECE_error_mukhoti as adaECE_error

# This is a simple function to call various other functions (test_classification_net_adafocal, expected_calibration_error, 
# ClasswiseECELoss, maximum_calibration_error, adaECE_error_mukhoti) defined in utils/metrics.py

def evaluate_dataset(model, dataloader, device, num_bins, num_labels):

    loss, confusion_matrix, acc, labels, predictions, confidences, logits = test_classification_net(model, dataloader, device)

    ece, bin_dict = expected_calibration_error(confidences, predictions, labels, num_bins=num_bins)
    adaece, adabin_dict = adaECE_error(confidences, predictions, labels, num_bins=num_bins)
    mce = maximum_calibration_error(confidences, predictions, labels, num_bins=num_bins)
    classwise_ece, per_class_ece = ClasswiseECELoss(n_bins=num_bins)(logits, torch.tensor(labels))
    
    return loss, confusion_matrix, acc, ece, bin_dict, adaece, adabin_dict, mce, classwise_ece, per_class_ece, logits, labels

def evaluate_dataset_train(labels, logits, predictions, confidences, num_bins):

    ece, bin_dict = expected_calibration_error(confidences, predictions, labels)
    adaece, adabin_dict = adaECE_error(confidences, predictions, labels)
    classwise_ece, per_class_ece = ClasswiseECELoss()(logits, torch.tensor(labels))
    return ece, bin_dict, adaece, adabin_dict, classwise_ece, per_class_ece