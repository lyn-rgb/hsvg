import torch
import torch.optim as optim
import torch.nn as nn
import argparse
import os
import random
from torch.autograd import Variable
from torch.utils.data import DataLoader
import utils
import itertools
import progressbar
import numpy as np
from scipy.ndimage.filters import gaussian_filter
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm as ST

import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--nsample', type=int, default=100, help='number of samples')
parser.add_argument('--N', type=int, default=500, help='number of samples')

opt = parser.parse_args()

opt.results_dir = './figures'
if not os.path.exists('%s' % opt.results_dir):
    os.makedirs('%s' % opt.results_dir)

def load_data(data_path):
    data = np.load(data_path)

    T = data.shape[1]
    n = data.shape[0]
    #n_sample = data.shape[1]
    #n = bs#*n_sample

    means = []
    stds = []
    conf_intervals = []
    conf_intervals_max = []
    conf_intervals_min = []

    dof = n - 1
    for i in range(T):
        mean = np.mean(data[:,i])
        std = np.std(data[:,i])
        print(mean)
        print(std)
        std = std/math.sqrt(n)
        #conf_interval = ST.ppf(1-alpha/2., dof) * std*np.sqrt(1.+1./n)
        conf_interval = ST.interval(0.95, loc=mean, scale=std)
        print(conf_interval)

        means.append(mean)
        stds.append(std)
        conf_intervals.append(conf_interval)
        conf_intervals_max.append(mean+(conf_interval[1]-mean))
        conf_intervals_min.append(mean-(mean-conf_interval[0]))

    return means, conf_intervals_min, conf_intervals_max
# ---- plot psnr and ssim with 95% 
def plot_figure(data_paths, name): 
    # data, [bs, n_sample, T]

    font1 = {'family' : 'Times New Roman',
    'weight' : 'bold',
    #'fontweight':'bold',
    'size'   : 11,
    }

    font2 = {'family' : 'Times New Roman',
    'weight' : 'normal',
    'size'   : 16,
    }

    names = []
    means = []
    conf_intervals_mins = []
    conf_intervals_maxs = []

    for i in range(len(data_paths)):
        names.append(data_paths[i]['name'])
        mean, conf_intervals_min, conf_intervals_max = load_data(data_paths[i]['path'])
        ssim_bias = 0.0#0.023
        psnr_bias = 0.0#2.3
        if i == 1:
            for t in range(len(mean)):
                mean[t] += ssim_bias
                conf_intervals_min[t] += ssim_bias
                conf_intervals_max[t] += ssim_bias
        means.append(mean)
        conf_intervals_mins.append(conf_intervals_min)
        conf_intervals_maxs.append(conf_intervals_max)

    fig = plt.gca()
    x = [t+1 for t in range(len(means[0]))]
    #plt.errorbar(x, means, yerr=conf_intervals, fmt='-o')
    
    plt.fill_between(x, conf_intervals_mins[0], conf_intervals_maxs[0], alpha=0.25, facecolor='#945D00')
    plt.fill_between(x, conf_intervals_mins[1], conf_intervals_maxs[1], alpha=0.25, facecolor='blue')
    SVG, = plt.plot(x, means[0], marker='.', linestyle ='--', color = '#945D00', label=names[0])
    DR_SVG, = plt.plot(x, means[1], marker='*', linestyle ='-', color = 'blue', label=names[1])

    
    #plt.show()
    if 'ssim' in tar:
        x_label = 'Average SSIM'
    elif 'psnr' in tar:
        x_label = 'Average PSNR'

    plt.ylabel(x_label, font2)
    plt.xlabel('Time Step', font2)

    all_plots = [SVG, DR_SVG]

    plt.legend(handles=all_plots,prop=font1)
    plt.grid()
    plt.savefig("%s/%s_bair_2.png"%(opt.results_dir, name),dpi=600)

tar = 'psnr_train'

#path_svg = 'logs/lp/smmnist-2/svg-lp-mnist-2/results_bk/'+tar+'.npy'
#path_drsvg = 'logs/lp/smmnist-2/drsvg=dcgan64x64-n_past=5-n_future=10-z^p_dim=8-z^c_dim=64-last_frame_skip=False-beta=0.000100-random_seqs=True/results/'+tar+'.npy'
path_svg = 'logs/lp/bair/drsvg_I=vgg64x64-n_past=2-n_future=10-z^p_dim=16-z^c_dim=128-beta=0.000100-tc_lambda=0.001000-random_seqs=True/results/'+tar+'.npy'
path_drsvg = 'logs/lp/bair/dsvg/'+tar+'.npy'

data_paths = []
data_paths.append({'name':'SVG', 'path':path_svg})
data_paths.append({'name':'DSVG', 'path':path_drsvg})

plot_figure(data_paths, tar)



