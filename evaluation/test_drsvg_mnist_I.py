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
import progressbar
import numpy as np
from scipy.ndimage.filters import gaussian_filter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
from scipy.stats import norm as ST

import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=10, type=int, help='batch size')
parser.add_argument('--data_root', default='data', help='root directory for data')
parser.add_argument('--model_path', default='', help='path to model')
parser.add_argument('--log_dir', default='logs/lp', help='directory to save generations to')
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
parser.add_argument('--factor', type=int, default=8, help='|z^c|/|z^p|')
parser.add_argument('--g_dim', type=int, default=128, help='dimensionality of encoder output vector and decoder input vector')
parser.add_argument('--num_digits', type=int, default=2, help='number of digits for moving mnist')

parser.add_argument('--channels', default=1, type=int)
parser.add_argument('--rnn_size', type=int, default=256, help='dimensionality of hidden layer')
parser.add_argument('--prior_rnn_layers', type=int, default=1, help='number of layers')
parser.add_argument('--posterior_rnn_layers', type=int, default=1, help='number of layers')
parser.add_argument('--predictor_rnn_layers', type=int, default=2, help='number of layers')
opt = parser.parse_args()


#name = 'drsvg_tc=%s%dx%d-n_past=5-n_future=10-z^p_dim=%d-z^c_dim=%d-last_frame_skip=False-beta=%.6f-random_seqs=True' % (opt.model, opt.image_width, opt.image_width, opt.z_dim, opt.z_dim*opt.factor, opt.beta)
name = 'drsvg=dcgan64x64-n_past=5-n_future=10-z^p_dim=8-z^c_dim=64-last_frame_skip=False-beta=0.000100-random_seqs=True'
if opt.dataset == 'smmnist':
    opt.log_dir = '%s/%s-%d/%s' % (opt.log_dir, opt.dataset, opt.num_digits, name)
else:
    opt.log_dir = '%s/%s/%s' % (opt.log_dir, opt.dataset, name)

opt.name = name

opt.dis_dir = 'logs/lp/smmnist-2/dsicriminator'
#opt.model_path = 'logs/lp/smmnist-2/vae-gan-v4_model=dcgan64x64-rnn_size=256-predictor-posterior-prior-rnn_layers=2-1-1-n_past=5-n_future=10-lr=0.0020-g_dim=128-z_dim=8-last_frame_skip=False-beta=0.000100/model.pth'
#opt.log_dir = 'logs/lp/smmnist-2/vae-gan-v4_model=dcgan64x64-rnn_size=256-predictor-posterior-prior-rnn_layers=2-1-1-n_past=5-n_future=10-lr=0.0020-g_dim=128-z_dim=8-last_frame_skip=False-beta=0.000100/lstm'
#opt.results_dir = 'logs/lp/smmnist-2/vae-gan-v4_model=dcgan64x64-rnn_size=256-predictor-posterior-prior-rnn_layers=2-1-1-n_past=5-n_future=10-lr=0.0020-g_dim=128-z_dim=8-last_frame_skip=False-beta=0.000100/results'

#os.makedirs('%s' % opt.log_dir, exist_ok=True)
if not os.path.exists('%s/samples' % opt.log_dir):
    os.makedirs('%s/samples' % opt.log_dir)
if not os.path.exists('%s/results' % opt.log_dir):
    os.makedirs('%s/results' % opt.log_dir)

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
tmp = torch.load('%s/opt.pth' % opt.log_dir)
import models.lstm as lstm_models
if opt.model == 'dcgan':
    if opt.image_width == 64:
        import models.dcgan_64 as model 
    elif opt.image_width == 128:
        import models.dcgan_128 as model  
elif opt.model == 'vgg':
    if opt.image_width == 64:
        import models.vgg_64 as model
    elif opt.image_width == 128:
        import models.vgg_128 as model
else:
    raise ValueError('Unknown model: %s' % opt.model)
# define
frame_predictor = lstm_models.lstm((opt.factor+1)*opt.z_dim, opt.g_dim, opt.rnn_size, opt.predictor_rnn_layers, int(opt.batch_size/len(opt.gpu_ids)))
posterior_pose = lstm_models.gaussian_lstm(opt.g_dim+opt.factor*opt.z_dim, opt.z_dim, opt.rnn_size, opt.posterior_rnn_layers, int(opt.batch_size/len(opt.gpu_ids)))
prior = lstm_models.gaussian_lstm(opt.g_dim+opt.factor*opt.z_dim, opt.z_dim, opt.rnn_size, opt.prior_rnn_layers, int(opt.batch_size/len(opt.gpu_ids)))

cont_encoder = model.cont_encoder(opt.z_dim*opt.factor, opt.channels*opt.n_past)  #g_dim = 64 or 128
pose_encoder = model.pose_encoder(opt.g_dim, opt.channels)
decoder = model.decoder(opt.g_dim, opt.channels)

# init
frame_predictor = utils.init_net(frame_predictor, init_type='normal', init_gain=0.02, gpu_ids=opt.gpu_ids)
posterior_pose = utils.init_net(posterior_pose, init_type='normal', init_gain=0.02, gpu_ids=opt.gpu_ids)
prior = utils.init_net(prior, init_type='normal', init_gain=0.02, gpu_ids=opt.gpu_ids)

cont_encoder = utils.init_net(cont_encoder, init_type='normal', init_gain=0.02, gpu_ids=opt.gpu_ids)
pose_encoder = utils.init_net(pose_encoder, init_type='normal', init_gain=0.02, gpu_ids=opt.gpu_ids)
decoder = utils.init_net(decoder, init_type='normal', init_gain=0.02, gpu_ids=opt.gpu_ids)

# load
utils.load_network(frame_predictor, 'frame_predictor', 'last', opt.log_dir,device)
utils.load_network(posterior_pose, 'posterior_pose', 'last', opt.log_dir,device)
utils.load_network(prior, 'prior', 'last', opt.log_dir,device)

utils.load_network(cont_encoder, 'cont_encoder', 'last', opt.log_dir,device)
utils.load_network(pose_encoder, 'pose_encoder', 'last', opt.log_dir,device)
utils.load_network(decoder, 'decoder', 'last', opt.log_dir,device)

frame_predictor.eval()
prior.eval()
posterior_pose.eval()

cont_encoder.eval()
pose_encoder.eval()
decoder.eval()

# -- discriminator
discriminator = model.discriminator(1)
discriminator = utils.init_net(discriminator, init_type='normal', init_gain=0.02, gpu_ids=[7])

#if opt.model_dir != '':
utils.load_network(discriminator, 'discriminator', 'last', opt.dis_dir, device)
discriminator.eval()

nets = [frame_predictor, posterior_pose, prior, cont_encoder, pose_encoder, decoder, discriminator]
# ---------------- discriminator ----------
utils.set_requires_grad(nets, False)

# ---------------- set the options ----------------
opt.dataset = tmp['opt'].dataset
opt.last_frame_skip = tmp['opt'].last_frame_skip
opt.channels = tmp['opt'].channels
opt.image_width = tmp['opt'].image_width

print(opt)

# --------- load a dataset ------------------------------------
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
    # get approx posterior sample
    posterior_gen = []
    posterior_gen.append(x[0])
    x_in = x[0]
    # ------------ calculate the content posterior
    xs = []
    for i in range(0, opt.n_past):
        xs.append(x[i])
    #if True:
    random.shuffle(xs)
    #xc = torch.cat(xs, 1)
    mu_c, logvar_c, skip = cont_encoder(torch.cat(xs, 1))
    mu_c = mu_c.detach()

    for i in range(1, opt.n_eval):
        h_target = pose_encoder(x[i]).detach()
        mu_t_p, logvar_t_p = posterior_pose(torch.cat([h_target, mu_c],1), time_step = i-1)
        z_t_p = utils.reparameterize(mu_t_p, logvar_t_p)
        if i < opt.n_past:
            frame_predictor(torch.cat([z_t_p, mu_c], 1), time_step = i-1) 
            posterior_gen.append(x[i])
            x_in = x[i]
        else:
            h_pred = frame_predictor(torch.cat([z_t_p, mu_c], 1), time_step = i-1).detach()
            x_in = decoder([h_pred, skip]).detach()
            posterior_gen.append(x_in)
  
    nsample = opt.nsample
    ssim = np.zeros((opt.batch_size, nsample, opt.n_future))
    psnr = np.zeros((opt.batch_size, nsample, opt.n_future))

    ccm_pred = np.zeros((opt.batch_size, nsample, opt.n_eval-1))
    ccm_gt = np.zeros((opt.batch_size, opt.n_eval-1))

    progress = progressbar.ProgressBar(maxval=nsample).start()
    all_gen = []
    for i in range(1, opt.n_eval):
        out_gt = discriminator(torch.cat([x[0], x[i]],dim=1))
        ccm_i_gt = out_gt.mean().data.cpu().numpy()
        print('time step %d, mean out gt: %.4f'%(i,ccm_i_gt))
        ccm_gt[:,i-1] = out_gt.squeeze().data.cpu().numpy()
    
    hs = []
    for i in range(0, opt.n_past):
        hs.append(pose_encoder(x[i]).detach())

    for s in range(nsample):
        progress.update(s+1)
        gen_seq = []
        gt_seq = []
        x_in = x[0]
        all_gen.append([])
        all_gen[s].append(x_in)

        h = pose_encoder(x[0]).detach()
        
        tic = time.time()
        for i in range(1, opt.n_eval):
            #if i < opt.n_past:
            #    h = hs[i-1].detach()
            #    h_target = hs[i].detach()
            #else:
            #h = pose_encoder(x_in).detach()
            h_target = pose_encoder(x[i]).detach()

            if i < opt.n_past:
                mu_t_p, logvar_t_p = posterior_pose(torch.cat([h_target, mu_c],1), time_step = i-1)
                z_t_p = utils.reparameterize(mu_t_p, logvar_t_p)
                prior(torch.cat([h, mu_c],1), time_step = i-1)
                frame_predictor(torch.cat([z_t_p, mu_c], 1), time_step = i-1)
                x_in = x[i]
                all_gen[s].append(x_in)
                h = h_target
            else:
                mu_t_pp, logvar_t_pp = prior(torch.cat([h, mu_c],1),time_step = i-1)
                z_t = utils.reparameterize(mu_t_pp, logvar_t_pp)
                h_pred = frame_predictor(torch.cat([z_t, mu_c], 1), time_step = i-1).detach()
                x_in = decoder([h_pred, skip]).detach()
                gen_seq.append(x_in.data.cpu().numpy())
                gt_seq.append(x[i].data.cpu().numpy())
                all_gen[s].append(x_in)
                h = pose_encoder(x_in).detach()                

            out_pred = discriminator(torch.cat([x[0],x_in],dim=1))
            #ccm_i_pred = out_pred.mean().data.cpu().numpy()
            #print('time step %d, mean out pred: %.4f'%(i,ccm_i_pred))
            ccm_pred[:, s, i-1] = out_pred.squeeze().data.cpu().numpy()
        toc = time.time()
        print('generate each sample including %d frames cost %f'%(opt.n_eval-1,toc-tic))
        _, ssim[:, s, :], psnr[:, s, :] = utils.eval_seq(gt_seq, gen_seq)

    progress.finish()
    utils.clear_progressbar()
    
    best_ssim = np.zeros((opt.batch_size, opt.n_future))
    best_psnr = np.zeros((opt.batch_size, opt.n_future))
    ###### ssim ######
    for i in range(opt.batch_size):
        gifs = [ [] for t in range(opt.n_eval) ]
        text = [ [] for t in range(opt.n_eval) ]

        mean_ssim = np.mean(ssim[i], 1)
        ordered = np.argsort(mean_ssim)
        best_ssim[i,:] = ssim[i,ordered[-1],:]

        mean_psnr = np.mean(psnr[i], 1)
        ordered_p = np.argsort(mean_psnr)
        best_psnr[i,:] = psnr[i, ordered_p[-1],:]

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

        fname = '%s/samples/%s_%d.gif' % (opt.log_dir, name, idx+i) 
        utils.save_gif_with_text(fname, gifs, text)


        # -- generate samples
        to_plot = []
        gts = []
        best_p = []
        rand_samples = [[] for s in range(len(rand_sidx))]
        for t in range(opt.n_eval):
            # gt
            gts.append(x[t][i])
            best_p.append(all_gen[ordered_p[-1]][t][i])

            # sample
            for s in range(len(rand_sidx)):
                rand_samples[s].append(all_gen[rand_sidx[s]][t][i])

        to_plot.append(gts)
        to_plot.append(best_p)
        for s in range(len(rand_sidx)):
            to_plot.append(rand_samples[s])
        fname = '%s/samples/%s_%d.png' % (opt.log_dir, name, idx+i)
        utils.save_tensors_image(fname, to_plot)

    return best_ssim, best_psnr, ccm_pred, ccm_gt

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
        print(mean)
        print(std)
        SE = std/math.sqrt(n)
        #conf_interval = ST.ppf(1-alpha/2., dof) * std*np.sqrt(1.+1./n)
        conf_interval = ST.interval(0.95, loc=mean, scale=SE)
        print(conf_interval)

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
    plt.savefig("%s/results/%s.png"%(opt.log_dir, name))


# ---- plot ccm
def plot_ccm(data_path, name): 
    # data, [N, n_sample, T]

    data = np.load(data_path)

    T = data.shape[2]
    n = data.shape[0]*data.shape[1]

    #n_sample = data.shape[1]
    #n = bs#*n_sample

    means = []
    stds = []
    conf_intervals = []
    conf_intervals_max = []
    conf_intervals_min = []

    dof = n - 1
    for i in range(T):
        mean = np.mean(data[:,:,i])
        std = np.std(data[:,:,i])
        print(mean)
        print(std)
        SE = std/math.sqrt(n)
        #conf_interval = ST.ppf(1-alpha/2., dof) * std*np.sqrt(1.+1./n)
        conf_interval = ST.interval(0.95, loc=mean, scale=SE)
        print(conf_interval)

        means.append(mean)
        stds.append(std)
        conf_intervals.append(conf_interval)
        conf_intervals_max.append(conf_interval[1])
        conf_intervals_min.append(conf_interval[0])

    fig = plt.gca()
    x = [t+1 for t in range(T)]
    #plt.errorbar(x, means, yerr=conf_intervals, fmt='-o')
    plt.plot(x, means, marker='o', linestyle ='-', color = 'blue', label='svg')
    #plt.fill_between(x, conf_intervals_min, conf_intervals_max, alpha=0.5, facecolor='#7EFF99')
    #plt.show()
    plt.savefig("%s/results/%s.png"%(opt.log_dir, name))


ssim_train = []
psnr_train = []
ssim_test = []
psnr_test = []

ccm_pred_train = []
ccm_pred_test = []
ccm_gt_train = []
ccm_gt_test = []

for i in range(0, opt.N, opt.batch_size):
    # plot train
    train_x = next(training_batch_generator)
    ssim, psnr, ccm_pred, ccm_gt = make_gifs(train_x, i, 'train')
    #ssim_train.append(ssim)
    #psnr_train.append(psnr)

    #ccm_pred_train.append(ccm_pred)
    #ccm_gt_train.append(ccm_gt)

    # plot test
    test_x = next(testing_batch_generator)
    ssim, psnr, ccm_pred, ccm_gt = make_gifs(test_x, i, 'test')
    #ssim_test.append(ssim)
    #psnr_test.append(psnr)

    #ccm_pred_test.append(ccm_pred)
    #ccm_gt_test.append(ccm_gt)

    print(i)
'''
ssim_train = np.concatenate(ssim_train, axis=0)
psnr_train = np.concatenate(psnr_train, axis=0)
ssim_test = np.concatenate(ssim_test, axis=0)
psnr_test = np.concatenate(psnr_test, axis=0)

ccm_pred_train = np.concatenate(ccm_pred_train, axis=0)
ccm_pred_test = np.concatenate(ccm_pred_test, axis=0)
ccm_gt_train = np.concatenate(ccm_gt_train, axis=0)
ccm_gt_test = np.concatenate(ccm_gt_test, axis=0)

print(ssim_train.shape)

np.save('%s/results/ssim_train'%(opt.log_dir),ssim_train)
np.save('%s/results/psnr_train'%(opt.log_dir),psnr_train)
np.save('%s/results/ssim_test'%(opt.log_dir),ssim_test)
np.save('%s/results/psnr_test'%(opt.log_dir),psnr_test)

np.save('%s/results/ccm_pred_train'%(opt.log_dir),ccm_pred_train)
np.save('%s/results/ccm_pred_test'%(opt.log_dir),ccm_pred_test)
np.save('%s/results/ccm_gt_train'%(opt.log_dir),ccm_gt_train)
np.save('%s/results/ccm_gt_test'%(opt.log_dir),ccm_gt_test)'''

#plot_figure('%s/results/ssim_train.npy'%(opt.log_dir), 'ssim_train')

#plot_ccm('%s/results/ccm_pred_train.npy'%(opt.log_dir), 'ccm_pred_train')

