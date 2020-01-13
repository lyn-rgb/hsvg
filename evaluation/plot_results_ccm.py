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

def load_data(data_path, ccm = False):
    data = np.load(data_path)

    if ccm:
        data = 2*data-1

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
def plot_figure(data_paths, name, ccm = False): 
    # data, [bs, n_sample, T]

    font1 = {'family' : 'Times New Roman',
    'weight' : 'bold',
    #'fontweight':'bold',
    'size'   : 12,
    }

    font2 = {'family' : 'Times New Roman',
    'weight' : 'normal',
    'size'   : 12,
    }

    names = []
    means = []
    conf_intervals_mins = []
    conf_intervals_maxs = []

        #plt.show()
    if 'acc' in tar:
        x_label = 'Average Accuracy'
    elif 'pa' in tar:
        x_label = 'Average Positive Activation'
    elif 'na' in tar:
        x_label = 'Average Negative Activation'

    # CCM
    if ccm:
        x_label = 'Content Consistency Measurement'

    for i in range(len(data_paths)):
        names.append(data_paths[i]['name'])
        mean, conf_intervals_min, conf_intervals_max = load_data(data_paths[i]['path'], ccm)
        if i == 2:
            bias = 0.0
            for t in range(len(mean)):
                mean[t] += bias
                conf_intervals_min[t] += bias
                conf_intervals_max[t] += bias
        means.append(mean)
        conf_intervals_mins.append(conf_intervals_min)
        conf_intervals_maxs.append(conf_intervals_max)

    fig = plt.gca()
    x = [t+1 for t in range(len(means[0]))]
    #plt.errorbar(x, means, yerr=conf_intervals, fmt='-o')
    plt.fill_between(x, conf_intervals_mins[0], conf_intervals_maxs[0], alpha=0.25, facecolor='green')
    plt.fill_between(x, conf_intervals_mins[1], conf_intervals_maxs[1], alpha=0.25, facecolor='#945D00')
    plt.fill_between(x, conf_intervals_mins[2], conf_intervals_maxs[2], alpha=0.25, facecolor='blue')

    GT, = plt.plot(x, means[0], marker='^', linestyle ='-.', color = 'green', label=names[0])
    SVG, = plt.plot(x, means[1], marker='.', linestyle ='--', color = '#945D00', label=names[1])
    DR_SVG, = plt.plot(x, means[2], marker='*', linestyle ='-', color = 'blue', label=names[2])


    plt.ylabel(x_label, font2)
    plt.xlabel('Time Step', font2)

    all_plots = [GT, SVG, DR_SVG]

    plt.legend(handles=all_plots,prop=font1)
    plt.grid()
    plt.savefig("%s/%s.png"%(opt.results_dir, name),dpi=600)

# --------------------- Plot time-wise ccm, ccm = 2*acc - 1
tar = 'epoch99_na'

path_gt = 'logs/lp/smmnist-2/discriminator_GT=dcgan64x64_sgd/results/'+tar+'.npy'
path_svg = 'logs/lp/smmnist-2/discriminator_SVG/results/'+tar+'.npy'
path_drsvg = 'logs/lp/smmnist-2/discriminator_DSVG=dcgan64x64/results/'+tar+'.npy'


data_paths = []
data_paths.append({'name':'G.T.', 'path':path_gt})
data_paths.append({'name':'SVG', 'path':path_svg})
data_paths.append({'name':'DSVG', 'path':path_drsvg})

plot_figure(data_paths, tar, False)

# --------------------- Plot epoch-wise ccm
'''path_gt = 'logs/lp/smmnist-2/discriminator_GT/results/'#+file_name+'.npy'
path_svg = 'logs/lp/smmnist-2/discriminator_SVG/results/'#+file_name+'.npy'
path_dsvg = 'logs/lp/smmnist-2/discriminator_DSVG=dcgan64x64/results/'#+file_name+'.npy'

def load_epoch_wise(data_path):
    means = []
    conf_intervals_mins = []
    conf_intervals_maxs = []
    for i in range(100):
        file_name = 'epoch%d_acc'%(i)
        file_path = data_path + file_name+'.npy'
        data = np.load(file_path)
        data = 2*data - 1
        data = np.concatenate(data, axis=0)

        mean = np.mean(data)
        std = np.std(data)
        n = data.shape[0]
        SE = std/math.sqrt(n)
        #conf_interval = ST.ppf(1-alpha/2., dof) * std*np.sqrt(1.+1./n)
        conf_interval = ST.interval(0.95, loc=mean, scale=SE)
        means.append(mean)
        conf_intervals_maxs.append(conf_interval[1])
        conf_intervals_mins.append(conf_interval[0])

    return means, conf_intervals_mins, conf_intervals_maxs

means_svg, conf_intervals_mins_svg, conf_intervals_maxs_svg = load_epoch_wise(path_svg)
means_dsvg, conf_intervals_mins_dsvg, conf_intervals_maxs_dsvg = load_epoch_wise(path_dsvg)

font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
#'fontweight':'bold',
'size'   : 12,
}

font2 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 12,
}

y_label = 'Content Consistency Measurement'

fig = plt.gca()
x = [t+1 for t in range(len(means_svg))]
    #plt.errorbar(x, means, yerr=conf_intervals, fmt='-o')
    
plt.fill_between(x, conf_intervals_mins_svg, conf_intervals_maxs_svg, alpha=0.25, facecolor='#945D00')
plt.fill_between(x, conf_intervals_mins_dsvg, conf_intervals_maxs_dsvg, alpha=0.25, facecolor='blue')
SVG, = plt.plot(x, means_svg, marker='.', linestyle ='--', color = '#945D00', label='SVG')
DR_SVG, = plt.plot(x, means_dsvg, marker='*', linestyle ='-', color = 'blue', label='DSVG')

plt.ylabel(y_label, font2)
plt.xlabel('Epoch', font2)

all_plots = [SVG, DR_SVG]

plt.legend(handles=all_plots,prop=font1)
plt.grid()
plt.savefig("%s/epoch_wise_ccm.png"%(opt.results_dir),dpi=600)'''




