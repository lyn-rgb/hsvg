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

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--data_root', default='data', help='root directory for data')
parser.add_argument('--model_path', default='', help='path to model')
parser.add_argument('--log_dir', default='', help='directory to save generations to')
parser.add_argument('--seed', default=1, type=int, help='manual seed')
parser.add_argument('--n_past', type=int, default=2, help='number of frames to condition on')
parser.add_argument('--n_future', type=int, default=28, help='number of frames to predict')
parser.add_argument('--num_threads', type=int, default=0, help='number of data loading threads')
parser.add_argument('--nsample', type=int, default=100, help='number of samples')
parser.add_argument('--N', type=int, default=256, help='number of samples')


opt = parser.parse_args()
#os.makedirs('%s' % opt.log_dir, exist_ok=True)

if not os.path.exists('%s/quality_results' % opt.log_dir):
os.makedirs('%s/quality_results' % opt.log_dir)
if not os.path.exists('%s/quantity_results' % opt.log_dir):
os.makedirs('%s/quantity_results' % opt.log_dir)

opt.n_eval = opt.n_past+opt.n_future
opt.max_step = opt.n_eval

print("Random Seed: ", opt.seed)
random.seed(opt.seed)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed_all(opt.seed)
dtype = torch.cuda.FloatTensor



# ---------------- load the models  ----------------
tmp = torch.load(opt.model_path)
frame_predictor = tmp['frame_predictor']
posterior = tmp['posterior']
prior = tmp['prior']
frame_predictor.eval()
prior.eval()
posterior.eval()
encoder = tmp['encoder']
decoder = tmp['decoder']
encoder.train()
decoder.train()
frame_predictor.batch_size = opt.batch_size
posterior.batch_size = opt.batch_size
prior.batch_size = opt.batch_size
opt.g_dim = tmp['opt'].g_dim
opt.z_dim = tmp['opt'].z_dim
opt.num_digits = tmp['opt'].num_digits


# --------- transfer to gpu ------------------------------------
frame_predictor.cuda()
posterior.cuda()
prior.cuda()
encoder.cuda()
decoder.cuda()

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
    frame_predictor.hidden = frame_predictor.init_hidden()
    posterior.hidden = posterior.init_hidden()
    posterior_gen = []
    posterior_gen.append(x[0])
    x_in = x[0]
    for i in range(1, opt.n_eval):
        h = encoder(x_in)
        h_target = encoder(x[i])[0].detach()
        if opt.last_frame_skip or i < opt.n_past:	
            h, skip = h
        else:
            h, _ = h
        h = h.detach()
        _, z_t, _= posterior(h_target) # take the mean
        if i < opt.n_past:
            frame_predictor(torch.cat([h, z_t], 1)) 
            posterior_gen.append(x[i])
            x_in = x[i]
        else:
            h_pred = frame_predictor(torch.cat([h, z_t], 1)).detach()
            x_in = decoder([h_pred, skip]).detach()
            posterior_gen.append(x_in)
  

    nsample = opt.nsample
    ssim = np.zeros((opt.batch_size, nsample, opt.n_future))
    psnr = np.zeros((opt.batch_size, nsample, opt.n_future))
    var_np = np.zeros((opt.batch_size, nsample, opt.n_future, opt.z_dim))
    progress = progressbar.ProgressBar(max_value=nsample).start()
    all_gen = []
    for s in range(nsample):
        progress.update(s+1)
        gen_seq = []
        gt_seq = []
        frame_predictor.hidden = frame_predictor.init_hidden()
        posterior.hidden = posterior.init_hidden()
        prior.hidden = prior.init_hidden()
        x_in = x[0]
        all_gen.append([])
        all_gen[s].append(x_in)
        for i in range(1, opt.n_eval):
            h = encoder(x_in)
            if opt.last_frame_skip or i < opt.n_past:	
                h, skip = h
            else:
                h, _ = h
            h = h.detach()
            if i < opt.n_past:
                h_target = encoder(x[i])[0].detach()
                z_t, _, _ = posterior(h_target)
                prior(h)
                frame_predictor(torch.cat([h, z_t], 1))
                x_in = x[i]
                all_gen[s].append(x_in)
            else:
                z_t, mu, logvar = prior(h)
                h = frame_predictor(torch.cat([h, z_t], 1)).detach()
                x_in = decoder([h, skip]).detach()
                gen_seq.append(x_in.data.cpu().numpy())
                gt_seq.append(x[i].data.cpu().numpy())
                all_gen[s].append(x_in)

                var = torch.exp(logvar)  # BxC
                var_np[:,s, i - opt.n_past,:] = var.data.cpu().numpy()

        _, ssim[:, s, :], psnr[:, s, :] = utils.eval_seq(gt_seq, gen_seq)

    progress.finish()
    utils.clear_progressbar()
    
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

        #fname = '%s/%s_%d.gif' % (opt.log_dir, name, idx+i) 
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
