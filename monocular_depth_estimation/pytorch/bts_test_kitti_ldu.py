# Copyright (C) 2019 Jin Han Lee
#
# This file is a part of BTS.
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>

from __future__ import absolute_import, division, print_function

import os
import argparse
import time
import numpy as np
import sys

import torch
from torch.autograd import Variable

from tqdm import tqdm

from bts_dataloader import *
from sparsification import sparsification_error_gpu

from bts_ldu import BtsModel

def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield arg


parser = argparse.ArgumentParser(description='BTS PyTorch implementation.', fromfile_prefix_chars='@')
parser.convert_arg_line_to_args = convert_arg_line_to_args

parser.add_argument('--model_name', type=str, help='model name', default='bts_nyu_v2')
parser.add_argument('--encoder', type=str, help='type of encoder, vgg or desenet121_bts or densenet161_bts',
                    default='densenet161_bts')
parser.add_argument('--data_path_eval', type=str, help='path to the data', required=True)
parser.add_argument('--gt_path_eval', type=str, help='path to the data', required=True)
parser.add_argument('--filenames_file_eval', type=str, help='path to the filenames text file', required=True)
parser.add_argument('--input_height', type=int, help='input height', default=480)
parser.add_argument('--input_width', type=int, help='input width', default=640)
parser.add_argument('--max_depth', type=float, help='maximum depth in estimation', default=80)

parser.add_argument('--checkpoint_path', type=str, help='path to a specific checkpoint to load', default='')

parser.add_argument('--dataset', type=str, help='dataset to train on, make3d or nyudepthv2', default='nyu')
parser.add_argument('--do_kb_crop', help='if set, crop input images as kitti benchmark images', action='store_true')
parser.add_argument('--garg_crop', help='if set, crops according to Garg  ECCV16', action='store_true')

parser.add_argument('--bts_size', type=int,   help='initial num_filters in bts', default=512)

parser.add_argument('--clip_gt', help='if set, clipping the ground truth to the min-max depth', action='store_true')
parser.add_argument('--min_depth_eval',      type=float, help='minimum depth for evaluation', default=1e-3)
parser.add_argument('--max_depth_eval',      type=float, help='maximum depth for evaluation', default=80)

parser.add_argument('--nb_proto', type=int,   help='initial num_proto in bts', default=30)

if sys.argv.__len__() == 2:
    arg_filename_with_prefix = '@' + sys.argv[1]
    args = parser.parse_args([arg_filename_with_prefix])
else:
    args = parser.parse_args()


def get_num_lines(file_path):
    f = open(file_path, 'r')
    lines = f.readlines()
    f.close()
    return len(lines)

def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    d1 = (thresh < 1.25).mean()
    d2 = (thresh < 1.25 ** 2).mean()
    d3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred)**2) / gt)

    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

    err = np.abs(np.log10(pred) - np.log10(gt))
    log10 = np.mean(err)

    return silog, log10, abs_rel, sq_rel, rmse, rmse_log, d1, d2, d3

inv_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.225]
)


def test():
    """Test function."""
    args.mode = 'online_eval'
    args.distributed = False

    dataloader = BtsDataLoader(args, 'online_eval')

    model = BtsModel(params=args)
    model = torch.nn.DataParallel(model)

    if os.path.exists(args.checkpoint_path):
        checkpoint = torch.load(args.checkpoint_path)
    else:
        print('Wrong checkpoint path. Exit.')
        exit()
    model.load_state_dict(checkpoint['model'])
    model.eval()
    model.cuda()

    num_params = sum([np.prod(p.size()) for p in model.parameters()])
    print("Total number of parameters: {}".format(num_params))

    num_test_samples = get_num_lines(args.filenames_file_eval)

    print('now testing {} files with {}'.format(num_test_samples, args.checkpoint_path))

    start_time = time.time()
    with torch.no_grad():
        num_samples = len(dataloader.data)
        print(num_samples)
        nb_valid = 0
        silog = np.zeros(num_samples, np.float32)
        log10 = np.zeros(num_samples, np.float32)
        rms = np.zeros(num_samples, np.float32)
        log_rms = np.zeros(num_samples, np.float32)
        abs_rel = np.zeros(num_samples, np.float32)
        sq_rel = np.zeros(num_samples, np.float32)
        d1 = np.zeros(num_samples, np.float32)
        d2 = np.zeros(num_samples, np.float32)
        d3 = np.zeros(num_samples, np.float32)

        hist_pred_rmses = 0
        hist_oracle_rmses = 0
        nb_remain_rmses = 0
        hist_pred_rmses = 0
        hist_oracle_rmses = 0
        nb_remain_rmses = 0
        ausc_rmse = np.zeros(num_samples, np.float32)

        hist_pred_absrels = 0
        hist_oracle_absrels = 0
        nb_remain_absrels = 0
        hist_pred_absrels = 0
        hist_oracle_absrels = 0
        nb_remain_absrels = 0
        ausc_absrel = np.zeros(num_samples, np.float32)

        spar_rmse = 0
        spar_absr = 0

        for i, sample in tqdm(enumerate(tqdm(dataloader.data))):
            
            is_valid = sample['has_valid_depth']
            if not is_valid: continue
            else: nb_valid += 1

            image = Variable(sample['image'].cuda())
            focal = Variable(sample['focal'].cuda())
            depth_gt = Variable(sample['depth'].cuda())

            # Predict
            depth_gt = depth_gt.cpu().numpy().squeeze()

            depth_est, uncertainty = model(image, focal)
            depth_est = depth_est.cpu().numpy().squeeze()
            uncertainty = uncertainty.cpu().numpy().squeeze()

            if args.clip_gt:
                valid_mask = np.logical_and(depth_gt > args.min_depth_eval, depth_gt < args.max_depth_eval)
            else:
                valid_mask = (depth_gt > args.min_depth_eval)

            # We are using online-eval here, and the following operation is to imitate the operation in the test case in the original work.
            if args.do_kb_crop:
                height, width = depth_gt.shape
                top_margin = int(height - 352)
                left_margin = int((width - 1216) / 2)
                pred_depth_uncropped = np.zeros((height, width), dtype=np.float32)
                pred_depth_uncropped[top_margin:top_margin + 352, left_margin:left_margin + 1216] = depth_est
                depth_est = pred_depth_uncropped

                pred_depth_uncropped = np.zeros((height, width), dtype=np.float32)
                pred_depth_uncropped[top_margin:top_margin + 352, left_margin:left_margin + 1216] = uncertainty
                uncertainty = pred_depth_uncropped

            if args.clip_gt:
                depth_est[depth_est < args.min_depth_eval] = args.min_depth_eval
                depth_est[depth_est > args.max_depth_eval] = args.max_depth_eval
                depth_est[np.isinf(depth_est)] = args.max_depth_eval

                depth_gt[np.isinf(depth_gt)] = args.max_depth_eval
                depth_gt[np.isnan(depth_gt)] = args.min_depth_eval

            if args.garg_crop:
                gt_height, gt_width = depth_gt.shape
                eval_mask = np.zeros(valid_mask.shape)
                eval_mask[int(0.40810811 * gt_height):int(0.99189189 * gt_height), int(0.03594771 * gt_width):int(0.96405229 * gt_width)] = 1

                valid_mask = np.logical_and(valid_mask, eval_mask)

            uncertainty = torch.tensor(uncertainty).cuda()
            depth_est = torch.tensor(depth_est).cuda()
            depth_gt = torch.tensor(depth_gt).cuda()
            valid_mask = torch.tensor(valid_mask).cuda()
            depth_abs_error = abs(depth_est - depth_gt)

            hist_pred_rmse, hist_oracle_rmse, nb_remain_rmse, ausc_rmse[i] = sparsification_error_gpu(unc_npy = uncertainty[valid_mask], err_npy = depth_abs_error[valid_mask], gt_npy = depth_gt[valid_mask], is_rmse = True)
            hist_pred_rmses += hist_pred_rmse
            hist_oracle_rmses += hist_oracle_rmse
            nb_remain_rmses += nb_remain_rmse
            spar_rmse += np.trapz((hist_pred_rmse - hist_oracle_rmse), x = list(np.arange(start=0.0, stop=1.0, step=(1/100))))

            hist_pred_absrel, hist_oracle_absrel, nb_remain_absrel, ausc_absrel[i] = sparsification_error_gpu(unc_npy = uncertainty[valid_mask], err_npy = depth_abs_error[valid_mask], gt_npy = depth_gt[valid_mask], is_rmse = False)
            hist_pred_absrels += hist_pred_absrel
            hist_oracle_absrels += hist_oracle_absrel
            nb_remain_absrels += nb_remain_absrel
            spar_absr += np.trapz((hist_pred_absrel - hist_oracle_absrel), x = list(np.arange(start=0.0, stop=1.0, step=(1/100))))

            depth_est = depth_est.cpu().numpy()
            depth_gt = depth_gt.cpu().numpy()
            valid_mask = valid_mask.cpu().numpy()
            silog[i], log10[i], abs_rel[i], sq_rel[i], rms[i], log_rms[i], d1[i], d2[i], d3[i] = compute_errors(depth_gt[valid_mask], depth_est[valid_mask])
            
    print(nb_valid)
    print("{:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}".format(
        'd1', 'd2', 'd3', 'AbsRel', 'SqRel', 'RMSE', 'RMSElog', 'SILog', 'log10'))
    print("{:7.3f}, {:7.3f}, {:7.3f}, {:7.3f}, {:7.3f}, {:7.3f}, {:7.3f}, {:7.3f}, {:7.3f}".format(
        d1.sum()/nb_valid, d2.sum()/nb_valid, d3.sum()/nb_valid,
        abs_rel.sum()/nb_valid, sq_rel.sum()/nb_valid, rms.sum()/nb_valid, 
        log_rms.sum()/nb_valid, silog.sum()/nb_valid, log10.sum()/nb_valid))


    hist_pred_rmses = hist_pred_rmses/nb_valid
    hist_oracle_rmses = hist_oracle_rmses/nb_valid
    nb_remain_rmses = nb_remain_rmses/nb_valid

    hist_pred_absrels = hist_pred_absrels/nb_valid
    hist_oracle_absrels = hist_oracle_absrels/nb_valid
    nb_remain_absrels = nb_remain_absrels/nb_valid

    spar_rmse = spar_rmse/nb_valid
    spar_absr = spar_absr/nb_valid
    
    # to verify that the averages obtained by the two different methods are consistent.
    print('ausc_rmse', np.trapz((hist_pred_rmses - hist_oracle_rmses), x = list(np.arange(start=0.0, stop=1.0, step=(1/100)))))
    print('ausc_abrel', np.trapz((hist_pred_absrels - hist_oracle_absrels), x = list(np.arange(start=0.0, stop=1.0, step=(1/100)))))

    print('ausc_rmse', spar_rmse)
    print('ausc_abrel', spar_absr)

    elapsed_time = time.time() - start_time
    print('Elapesed time: %s' % str(elapsed_time))
    print('Done.')
    

if __name__ == '__main__':
    test()