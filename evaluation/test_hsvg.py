import torch
import torch.optim as optim
import torch.nn as nn
import argparse
import os
import random
import math
from torch.autograd import Variable
from torch.utils.data import DataLoader
import utils
import itertools
#import progressbar
from tqdm import tqdm

import numpy as np
from scipy.ndimage.filters import gaussian_filter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
from scipy.stats import norm as ST

from models.hsvgnet import hsvgnet as HSVG

import warnings
warnings.filterwarnings('ignore')

print('Testing ==>')
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=10, type=int, help='batch size')
parser.add_argument('--data_root', default='data', help='root directory for data')
parser.add_argument('--model_path', default='', help='path to model')
parser.add_argument('--log_dir', default='logs/hsvg/', help='directory to save generations to')
parser.add_argument('--seed', default=1, type=int, help='manual seed')
parser.add_argument('--n_past', type=int, default=5, help='number of frames to condition on')
parser.add_argument('--n_future', type=int, default=25, help='number of frames to predict')
parser.add_argument('--num_threads', type=int, default=5, help='number of data loading threads')
parser.add_argument('--nsample', type=int, default=100, help='number of samples')
parser.add_argument('--N', type=int, default=100, help='number of samples')
parser.add_argument('--gpus', default='7', help='multiple gpus' )

parser.add_argument('--model', default='dcgan', help='model type (dcgan | vgg)')
parser.add_argument('--image_width', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--dataset', default='smmnist', help='dataset to train with')
parser.add_argument('--z_dim', type=int, default=8, help='dimensionality of z^p_t')
parser.add_argument('--g_dim', type=int, default=128, help='dimensionality of encoder output vector and decoder input vector')
parser.add_argument('--num_digits', type=int, default=2, help='number of digits for moving mnist')

parser.add_argument('--channels', default=1, type=int)
parser.add_argument('--rnn_size', type=int, default=256, help='dimensionality of hidden layer')
parser.add_argument('--prior_rnn_layers', type=int, default=1, help='number of layers')
parser.add_argument('--posterior_rnn_layers', type=int, default=1, help='number of layers')
parser.add_argument('--predictor_rnn_layers', type=int, default=2, help='number of layers')
opt = parser.parse_args()

print(opt)

opt.log_dir = opt.log_dir + opt.dataset + '/' + opt.model_path
print('Model Path is [%s] ==>'%(opt.log_dir))

if not os.path.exists('%s/quality_results' % opt.log_dir):
os.makedirs('%s/quality_results' % opt.log_dir)
if not os.path.exists('%s/quantity_results' % opt.log_dir):
os.makedirs('%s/quantity_results' % opt.log_dir)

opt.n_eval = opt.n_past+opt.n_future
opt.max_step = opt.n_eval

gpus = opt.gpus.split(',')
gpus = [int(gpu_id) for gpu_id in gpus]
opt.gpu_ids = gpus

device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')
torch.cuda.set_device(opt.gpu_ids[0])

print("Random Seed: ", opt.seed)
random.seed(opt.seed)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed_all(opt.seed)
dtype = torch.cuda.FloatTensor

# ---------------- load the models  ----------------
print('Define HSVGNet ==>')
model = {'in_channels': opt.channels,
        'in_size': opt.image_width,
        'model_type': opt.model,
        'nlevel': opt.n_level,
        'z_dims': [opt.z_dim*(opt.n_level-i) for i in range(opt.n_level)],
        'o_dims': [(opt.g_dim//8)*(opt.n_level-i) for i in range(opt.n_level)],
        'g_dim': opt.g_dim,
        'rnn_size': [opt.rnn_size for i in range(opt.n_level)], 
        'rnnlayers':[opt.prior_rnn_layers for i in range(opt.n_level)],
        'rec_criterion': None,
        'kld_criterion': None,
    }
hsvg_net = HSVG(model)


print('Loading checkpoint ==>')
checkpoint_list = sorted(glob.glob('{}/hsvgnet_ep*.pth.tar'.format(checkpoint_dir)))
if checkpoint_list:
checkpoint_path = checkpoint_list[-1]
checkpoint_dict = torch.load(checkpoint_path)
#start_epoch = checkpoint_dict['epoch'] + 1
hsvg_net.load_state_dict(checkpoint_dict['hsvg_net'])
#hsvg_optimizer.load_state_dict(checkpoint_dict['hsvg_optimizer'])
print('Found checkpoint file {} ==>'.format(checkpoint_path))
else:
print('No matching checkpoint file found. Error occured. -_-!!!')

hsvg_net.eval()
hsvg_net.encoder.eval()
hsvg_net.decoder.eval()

hsvg_net.cuda()

# --------- load a dataset ------------------------------------
print('Constructing dataloader ==>')
train_data, test_data = utils.load_dataset(opt)

train_loader = DataLoader(train_data,
                        num_workers=opt.num_threads,
                        batch_size=opt.batch_size,
                        shuffle=False,
                        drop_last=True,
                        pin_memory=True)
test_loader = DataLoader(test_data,
                        num_workers=opt.num_threads,
                        batch_size=opt.batch_size,
                        shuffle=False,
                        drop_last=True,
                        pin_memory=True)

def get_training_batch():
while True:
    for sequence in train_loader:
        batch = utils.normalize_data(opt, dtype, sequence)
        yield batch
training_batch_generator = get_training_batch()

def get_testing_batch():
while True:
    for sequence in test_loader:
        batch = utils.normalize_data(opt, dtype, sequence)
        yield batch 
testing_batch_generator = get_testing_batch()

# --------- eval funtions ------------------------------------
def make_gifs(x, idx, name):
    # sample from approx posterior
    posterior_gen = []
    posterior_gen.append(x[0])
    x_in = x[0]
    
    # rec
    hsvg_net.init_states(x[0])
    for i in range(1, opt.n_eval):
        if i < opt.n_past:
            hs_rec, feats, zs, mus, logvars = hsvg_net.reconstruction(x[i])
            hsvg_net.skips = feats
            posterior_gen.append(x[i])
        else:
            hs_rec, feats, zs, mus, logvars = hsvg_net.reconstruction(x[i])
            x_rec = hsvg_net.decoding(hs_rec)
            posterior_gen.append(x_rec)
    
    # sample from prior
    nsample = opt.nsample
    ssim = np.zeros((opt.batch_size, nsample, opt.n_future))
    psnr = np.zeros((opt.batch_size, nsample, opt.n_future))
    # variance
    var_np = np.zeros((opt.batch_size, nsample, opt.n_future, opt.z_dim))
    all_gen = []
    gt_seq = [x[i].data.cpu().numpy() for i in range(opt.n_past, opt.n_eval)]

    for s in tqdm(range(nsample)):        
        gen_seq = []
        x_in = x[0]
        all_gen.append([])
        all_gen[s].append(x_in)
        
        hsvg_net.init_states(x[0])
        for i in range(1, opt.n_eval):
            if i < opt.n_past:
                hs_rec, feats, zs, mus, logvars = hsvg_net.reconstruction(x[i])
                hsvg_net.skips = feats
                all_gen[s].append(x[i])
            else:
                x_pred = hsvg_net.inference()
                gen_seq.append(x_pred.data.cpu().numpy())
                all_gen[s].append(x_pred)
                
                logvar = torch.cat(hsvg_net.logvars_prior, -1)
                var = torch.exp(logvar)  # BxC
                var_np[:,s, i - opt.n_past,:] = var.data.cpu().numpy()

        _, ssim[:, s, :], psnr[:, s, :] = utils.eval_seq(gt_seq, gen_seq)
    
    best_ssim = np.zeros((opt.batch_size, opt.n_future))
    best_psnr = np.zeros((opt.batch_size, opt.n_future))

    best_ssim_var = np.zeros((opt.batch_size, opt.n_future, opt.z_dim))
    best_psnr_var = np.zeros((opt.batch_size, opt.n_future, opt.z_dim))
    ###### ssim ######
    for i in range(opt.batch_size):
        gifs = [ [] for t in range(opt.n_eval) ]
        text = [ [] for t in range(opt.n_eval) ]

        mean_ssim = np.mean(ssim[i], 1)
        ordered = np.argsort(mean_ssim)
        best_ssim[i,:] = ssim[i,ordered[-1],:]
        # best ssim var
        best_ssim_var[i,:,:] = var_np[i, ordered[-1], :, :]

        mean_psnr = np.mean(psnr[i], 1)
        ordered_p = np.argsort(mean_psnr)
        best_psnr[i,:] = psnr[i, ordered_p[-1],:]
        # best psnr var
        best_psnr_var[i,:,:] = var_np[i, ordered_p[-1], :, :]

        rand_sidx = [np.random.randint(nsample) for s in range(3)]

        # -- generate gifs
        for t in range(opt.n_eval):
            # gt 
            gifs[t].append(add_border(x[t][i], 'green'))
            text[t].append('Ground\ntruth')
            #posterior 
            if t < opt.n_past:
                color = 'green'
            else:
                color = 'red'
            gifs[t].append(add_border(posterior_gen[t][i], color))
            text[t].append('Approx.\nposterior')
            # best 
            if t < opt.n_past:
                color = 'green'
            else:
                color = 'red'
            sidx = ordered[-1]
            gifs[t].append(add_border(all_gen[sidx][t][i], color))
            text[t].append('Best SSIM')
            # random 3
            for s in range(len(rand_sidx)):
                gifs[t].append(add_border(all_gen[rand_sidx[s]][t][i], color))
                text[t].append('Random\nsample %d' % (s+1))

        fname = '%s/quality_results/%s_%d.gif' % (opt.log_dir, name, idx+i) 
        utils.save_gif_with_text(fname, gifs, text)

        # -- generate samples
        to_plot = []
        gts = []
        best_s = []
        best_p = []
        rand_samples = [[] for s in range(len(rand_sidx))]
        for t in range(opt.n_eval):
            # gt
            gts.append(x[t][i])
            best_s.append(all_gen[ordered[-1]][t][i])
            best_p.append(all_gen[ordered_p[-1]][t][i])

            # sample
            for s in range(len(rand_sidx)):
                rand_samples[s].append(all_gen[rand_sidx[s]][t][i])

        to_plot.append(gts)
        to_plot.append(best_s)
        to_plot.append(best_p)
        for s in range(len(rand_sidx)):
            to_plot.append(rand_samples[s])
        fname = '%s/quality_results/%s_%d.png' % (opt.log_dir, name, idx+i)
        utils.save_tensors_image(fname, to_plot)

    return best_ssim, best_psnr, best_ssim_var, best_psnr_var

def add_border(x, color, pad=1):
    w = x.size()[1]
    nc = x.size()[0]
    px = Variable(torch.zeros(3, w+2*pad+30, w+2*pad))
    if color == 'red':
        px[0] =0.7 
    elif color == 'green':
        px[1] = 0.7
    if nc == 1:
        for c in range(3):
            px[c, pad:w+pad, pad:w+pad] = x
    else:
        px[:, pad:w+pad, pad:w+pad] = x
    return px


# ---- plot psnr and ssim with 95% 
def plot_figure(data_path, name, alpha=0.05): 
    # data, [bs, n_sample, T]

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
        #print(mean)
        #print(std)
        SE = std/math.sqrt(n)
        conf_interval = ST.interval(0.95, loc=mean, scale=SE)
        #print(conf_interval)

        means.append(mean)
        stds.append(std)
        conf_intervals.append(conf_interval)
        conf_intervals_max.append(conf_interval[1]/1.0)
        conf_intervals_min.append(conf_interval[0]/1.0)

    fig = plt.gca()
    x = [t+1 for t in range(T)]
    #plt.errorbar(x, means, yerr=conf_intervals, fmt='-o')
    plt.plot(x, means, marker='o', linestyle ='-', color = 'blue', label='svg')
    plt.fill_between(x, conf_intervals_min, conf_intervals_max, edgecolor='blue', facecolor='#7EFF99')
    #plt.show()
    plt.savefig("%s/quantity_results/%s.png"%(opt.log_dir, name))

def main():
    #ssim_train = []
    #psnr_train = []
    ssim_test = []
    psnr_test = []

    ssim_var_test = []
    psnr_var_test = []

    print('Start Testing ==>')
    for i in range(0, opt.N, opt.batch_size):
        # plot train
        '''
        train_x = next(training_batch_generator)
        ssim, psnr, ccm_pred, ccm_gt = make_gifs(train_x, i, 'train')
        ssim_train.append(ssim)
        psnr_train.append(psnr)
        '''

        # plot test
        test_x = next(testing_batch_generator)
        ssim, psnr, ssim_var, psnr_var = make_gifs(test_x, i, 'test')
        ssim_test.append(ssim)
        psnr_test.append(psnr)

        ssim_var_test.append(ssim_var)
        psnr_var_test.append(psnr_var)

        print('%d-th video clip was tested <=='%(i))

    #ssim_train = np.concatenate(ssim_train, axis=0)
    #psnr_train = np.concatenate(psnr_train, axis=0)
    ssim_test = np.concatenate(ssim_test, axis=0)
    psnr_test = np.concatenate(psnr_test, axis=0)
    
    ssim_var_test = np.concatenate(ssim_var_test, axis=0)
    psnr_var_test = np.concatenate(psnr_var_test, axis=0)
    
    #print(ssim_train.shape)
    #np.save('%s/quantity_results/ssim_train'%(opt.log_dir),ssim_train)
    #np.save('%s/quantity_results/psnr_train'%(opt.log_dir),psnr_train)
    np.save('%s/quantity_results/ssim_test_%d'%(opt.log_dir, opt.N),ssim_test)
    np.save('%s/quantity_results/psnr_test_%d'%(opt.log_dir, opt.N),psnr_test)

    np.save('%s/quantity_results/ssim_var_test_%d'%(opt.log_dir, opt.N),ssim_var_test)
    np.save('%s/quantity_results/psnr_var_test_%d'%(opt.log_dir, opt.N),psnr_var_test)

    plot_figure('%s/quantity_results/ssim_test.npy'%(opt.log_dir), 'ssim_test_%d'%(opt.N))

if __name__=='__main__':
    with torch.no_grad():
        main()
