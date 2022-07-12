import numpy as np
import torch

"""Calculate the sparsification error.

    Calcualte the sparsification error for a given array according to a reference array.

    Args:
        unc_npy: Flatten estimated uncertainty numpy array. 
        err_npy: Flatten ground truth error numpy array (abs error). 
        nb_bins: Number of bins using for uncertainty estimation. Each time, 1/nb_bins * 100% items with highest value will be removed.
        return_hist: if return histograms for drawing the sparsification curve, otherwise, directly return the sum of sparsification error.
        gt_npy: if calculate the AUSE accroding to Absrel or RMSE. By default, None, which indicates we calculate AUSE_RMSE
    Returns:
        By default, sum of the sparsification error after removing all the items in two given vectors given nb_bins.
        Given return_hist = True, three arrays corresponding to the components of sparsification curve.

"""
def sparsification_error_gpu(unc_npy, err_npy, nb_bins = 100, return_hist=True, gt_npy = None, is_rmse = True):

    hist_pred = []
    hist_oracle = []
    nb_remain = []

    # From small to big
    argsorted_U = torch.argsort(unc_npy)
    
    if not is_rmse:
        err_npy = err_npy/gt_npy
    else:
        err_npy = err_npy**2

    argsorted_E = torch.argsort(err_npy)

    total_len = len(unc_npy)

    sigma_pred_curves = []
    error_curves = []
    fractions = list(torch.arange(start=0.0, end=1.0, step=(1/nb_bins)))
    for fraction in fractions:
        if is_rmse:
            sigma_pred_curve = torch.mean(err_npy[argsorted_U[0:int((1.0-fraction)*total_len)]])
            error_curve = torch.mean(err_npy[argsorted_E[0:int((1.0-fraction)*total_len)]])
            sigma_pred_curve = torch.sqrt(sigma_pred_curve)
            error_curve = torch.sqrt(error_curve)
        else:
            sigma_pred_curve = torch.mean(err_npy[argsorted_U[0:int((1.0-fraction)*total_len)]])
            error_curve = torch.mean(err_npy[argsorted_E[0:int((1.0-fraction)*total_len)]])

        sigma_pred_curves.append(sigma_pred_curve)
        error_curves.append(error_curve)
        nb_remain.append(int((1.0-fraction)*total_len))
    
    hist_oracle = torch.tensor(error_curves)/error_curves[0].cpu()
    hist_pred = torch.tensor(sigma_pred_curves)/sigma_pred_curves[0].cpu()
    nb_remain = torch.tensor(nb_remain)
    sparsification_errors_pred = torch.trapz((hist_pred - hist_oracle), torch.arange(start=0.0, end=1.0, step=(1/nb_bins)))

    # without normalization. in our paper we use the codes shown above.
    # hist_oracle = torch.tensor(error_curves)
    # hist_pred = torch.tensor(sigma_pred_curves)
    # nb_remain = torch.tensor(nb_remain)
    # sparsification_errors_pred = torch.trapz((hist_pred), torch.arange(start=0.0, end=1.0, step=(1/nb_bins))) - torch.trapz((hist_oracle), torch.arange(start=0.0, end=1.0, step=(1/nb_bins)))

    if return_hist:
        return hist_pred, hist_oracle, nb_remain, sparsification_errors_pred
    else:
        return sparsification_errors_pred