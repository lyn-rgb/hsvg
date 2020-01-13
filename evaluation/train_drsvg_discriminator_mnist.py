# -*-coding:UTF-8_*_
#改变代码332
import torch
import torch.optim as optim
import torch.nn as nn
import argparse
import os
import random
from torch.autograd import Variable
from torch.utils.data import DataLoader
import utils
from utils import init_net, update_learning_rate, get_scheduler, save_network, load_network, print_network, reparameterize, set_requires_grad
import itertools
#import progressbar
from progressbar import *
import time
import numpy as np
#from models.image_pool import ImagePool, LatentPool
#from models.model import GANLoss

import warnings
from tensorboardX import SummaryWriter


warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=0.002, type=float, help='learning rate')
parser.add_argument('--beta1', default=0.9, type=float, help='momentum term for adam')
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--log_dir', default='logs/lp', help='base directory to save logs')
parser.add_argument('--model_dir', default='', help='base directory to save logs')   #save logs
parser.add_argument('--name', default='', help='identifier for directory')
parser.add_argument('--data_root', default='data', help='root directory for data')
parser.add_argument('--optimizer', default='adam', help='optimizer to train with')
parser.add_argument('--niter', type=int, default=100, help='number of epochs to train for')

parser.add_argument('--niter_decay', type=int, default=50, help='# of iter to linearly decay learning rate to zero')

parser.add_argument('--seed', default=1, type=int, help='manual seed')
parser.add_argument('--epoch_size', type=int, default=300, help='epoch size')

parser.add_argument('--image_width', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--channels', default=1, type=int)
parser.add_argument('--dataset', default='smmnist', help='dataset to train with')
parser.add_argument('--n_past', type=int, default=5, help='number of frames to condition on')
parser.add_argument('--n_future', type=int, default=25, help='number of frames to predict during training')
parser.add_argument('--n_eval', type=int, default=30, help='number of frames to predict during eval')

parser.add_argument('--z_dim', type=int, default=8, help='dimensionality of z^p_t')
parser.add_argument('--factor', type=int, default=8, help='|z^c|/|z^p|')
parser.add_argument('--g_dim', type=int, default=128, help='dimensionality of encoder output vector and decoder input vector')

parser.add_argument('--model', default='dcgan', help='model type (dcgan | vgg)')
parser.add_argument('--data_threads', type=int, default=5, help='number of data loading threads')
parser.add_argument('--num_digits', type=int, default=2, help='number of digits for moving mnist')

parser.add_argument('--lr_policy', type=str, default='step', help='learning rate policy. [linear | step | plateau | cosine]')
parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')

parser.add_argument('--rnn_size', type=int, default=256, help='dimensionality of hidden layer')
parser.add_argument('--prior_rnn_layers', type=int, default=1, help='number of layers')
parser.add_argument('--posterior_rnn_layers', type=int, default=1, help='number of layers')
parser.add_argument('--predictor_rnn_layers', type=int, default=2, help='number of layers')

#parser.add_argument('--gpu', type=int, default=3, help='gpu ids')
parser.add_argument('--gpus', default='2', help='multiple gpus' )
parser.add_argument('--svg_name', type = str, default='GT', help='SVG | DSVG' )

opt = parser.parse_args()
if opt.model_dir != '':
    # load model and continue training from checkpoint
    saved_model = torch.load('%s/discriminator_opt.pth' % opt.model_dir)
    #saved_opt = torch.load('%s/opt.pth' % opt.log_dir)
    optimizer = opt.optimizer
    model_dir = opt.model_dir
    opt = saved_opt['opt']
    opt.optimizer = optimizer
    opt.model_dir = model_dir
    opt.log_dir = '%s/continued' % opt.log_dir
else:
    name = 'discriminator_%s=%s%dx%d_sgd' % (opt.svg_name, opt.model, opt.image_width, opt.image_width)
    if opt.dataset == 'smmnist':
        opt.log_dir = '%s/%s-%d/%s' % (opt.log_dir, opt.dataset, opt.num_digits, name)
    else:
        opt.log_dir = '%s/%s/%s' % (opt.log_dir, opt.dataset, name)

    opt.name = name

if not os.path.exists('%s/' % opt.log_dir):
    os.makedirs('%s/' % opt.log_dir)

if not os.path.exists('%s/results' % opt.log_dir):
    os.makedirs('%s/results' % opt.log_dir)



run_forder = opt.dataset
if opt.dataset == 'smmnist':
    run_forder += '-%d' % opt.num_digits

writer_train = SummaryWriter('./runs/%s/%s/train/' %(run_forder, opt.name))
writer_test = SummaryWriter('./runs/%s/%s/test/' % (run_forder, opt.name))

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

gpus = opt.gpus.split(',')
gpus = [int(gpu_id) for gpu_id in gpus]
opt.gpu_ids = gpus #[opt.gpu]
#torch.distributed.init_process_group(backend="nccl")
#模型并行化

device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')
torch.cuda.set_device(opt.gpu_ids[0])

print("Random Seed: ", opt.seed)
random.seed(opt.seed)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed_all(opt.seed)
dtype = torch.cuda.FloatTensor
# ---------------- load the models  ----------------
print(opt)

# ------- load models of various svg
if opt.svg_name == 'GT':
    print('Training discriminator from the g.t.!')
# ------- load SVG
elif opt.svg_name == 'SVG':
    #import svg_models as models
    opt.model_path = 'logs/lp/smmnist-2/model=dcgan64x64-rnn_size=256-predictor-posterior-prior-rnn_layers=2-1-1-n_past=5-n_future=10-lr=0.0020-g_dim=128-z_dim=10-last_frame_skip=False-beta=0.0001000/model.pth'
    #opt.model_path = 'logs/lp/smmnist-2/svg-lp-mnist-2/model.pth'
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

    nets = [frame_predictor, posterior, prior, encoder, decoder]
    # ---------------- discriminator ----------
    utils.set_requires_grad(nets, False)

#-------- load DSVG
elif opt.svg_name == 'DSVG':
    opt.model_path = 'logs/lp/smmnist-2/drsvg=dcgan64x64-n_past=5-n_future=10-z^p_dim=8-z^c_dim=64-last_frame_skip=False-beta=0.000100-random_seqs=True'
    
    tmp = torch.load('%s/opt.pth' % opt.model_path)
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
    utils.load_network(frame_predictor, 'frame_predictor', 'last', opt.model_path,device)
    utils.load_network(posterior_pose, 'posterior_pose', 'last', opt.model_path,device)
    utils.load_network(prior, 'prior', 'last', opt.model_path,device)

    utils.load_network(cont_encoder, 'cont_encoder', 'last', opt.model_path,device)
    utils.load_network(pose_encoder, 'pose_encoder', 'last', opt.model_path,device)
    utils.load_network(decoder, 'decoder', 'last', opt.model_path,device)

    frame_predictor.eval()
    prior.eval()
    posterior_pose.eval()

    cont_encoder.eval()
    pose_encoder.eval()
    decoder.eval()

    nets = [frame_predictor, posterior_pose, prior, cont_encoder, pose_encoder, decoder]
    # ---------------- discriminator ----------
    utils.set_requires_grad(nets, False)

else:
    raise ValueError('Unknown svg model: %s' % opt.svg_name)
# -----------------------------------

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
       
discriminator = model.discriminator(opt.channels)
discriminator = init_net(discriminator, init_type='normal', init_gain=0.02, gpu_ids=opt.gpu_ids)

if opt.model_dir != '':
    load_network(discriminator, 'discriminator', 'last', opt.log_dir,device)

# ---------------- optimizers ----------------
#discriminator_optimizer = opt.optimizer(discriminator.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
if opt.optimizer == 'adam':
    opt.optimizer = optim.Adam
    discriminator_optimizer = opt.optimizer(discriminator.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
elif opt.optimizer == 'rmsprop':
    opt.optimizer = optim.RMSprop
    discriminator_optimizer = opt.optimizer(discriminator.parameters(), lr=opt.lr, alpha=0.99)
elif opt.optimizer == 'sgd':
    opt.optimizer = optim.SGD
    discriminator_optimizer = opt.optimizer(discriminator.parameters(), lr=opt.lr, momentum = 0.9)
else:
    raise ValueError('Unknown optimizer: %s' % opt.optimizer)

optimizers = []
optimizers.append(discriminator_optimizer)

# --------- print networks
print('###################################### Networks #######################################')
print('--------------------------- Discriminator (latent) -----------------------------')
print_network(discriminator,'discriminator')
print('###################################### end #######################################')

# --------- loss functions ------------------------------------

bce_criterion = nn.BCELoss()
bce_criterion.to(opt.gpu_ids[0])

# --------- load a dataset ------------------------------------
train_data, test_data = utils.load_dataset(opt)

train_loader = DataLoader(train_data,
                          num_workers=opt.data_threads,
                          batch_size=opt.batch_size,
                          shuffle=True,
                          drop_last=True,
                          pin_memory=True)
test_loader = DataLoader(test_data,
                         num_workers=opt.data_threads,
                         batch_size=opt.batch_size,
                         shuffle=True,
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

#--------- generate data with svg or drsvg
def generate_sequence(x):
    if opt.svg_name == 'GT':
        return x
    elif opt.svg_name == 'SVG':
        gen_seq = []
        frame_predictor.hidden = frame_predictor.init_hidden()
        posterior.hidden = posterior.init_hidden()
        prior.hidden = prior.init_hidden()
        x_in = x[0]
        gen_seq.append(x_in)
        for i in range(1, opt.n_eval):
            h = encoder(x_in)
            if i < opt.n_past:   
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
                gen_seq.append(x_in)
            else:
                z_t, _, _ = prior(h)
                h = frame_predictor(torch.cat([h, z_t], 1)).detach()
                x_in = decoder([h, skip]).detach()
                gen_seq.append(x_in)

        return gen_seq

    elif opt.svg_name == 'DSVG':
        gen_seq = []
        x_in = x[0]
        xs = []
        for i in range(0, opt.n_past):
            xs.append(x[i])
    
        random.shuffle(xs)
    
        mu_c, logvar_c, skip = cont_encoder(torch.cat(xs, 1))
        mu_c = mu_c.detach()
        
        gen_seq.append(x_in)
        h = pose_encoder(x[0]).detach()
        for i in range(1, opt.n_eval):
            h_target = pose_encoder(x[i]).detach()
            if i < opt.n_past:
                mu_t_p, logvar_t_p = posterior_pose(torch.cat([h_target, mu_c],1), time_step = i-1)
                z_t_p = utils.reparameterize(mu_t_p, logvar_t_p)
                prior(torch.cat([h, mu_c],1), time_step = i-1)
                frame_predictor(torch.cat([z_t_p, mu_c], 1), time_step = i-1)
                x_in = x[i]
                gen_seq.append(x_in)
                h = h_target
            else:
                mu_t_pp, logvar_t_pp = prior(torch.cat([h, mu_c],1),time_step = i-1)
                z_t = utils.reparameterize(mu_t_pp, logvar_t_pp)
                h_pred = frame_predictor(torch.cat([z_t, mu_c], 1), time_step = i-1).detach()
                x_in = decoder([h_pred, skip]).detach()
                gen_seq.append(x_in)
                h = pose_encoder(x_in).detach()

        return gen_seq
    else:
        raise ValueError('Unknown svg model: %s' % opt.svg_name)

# -- train
def train(x):
    discriminator.zero_grad()

    idx = random.randint(opt.n_past, opt.n_eval-1)
    x1 = x[0]
    x2 = x[idx]

    bs = x1.size(0)

    target = torch.cuda.FloatTensor(bs, 1)

    half = int(bs/2)
    rp = torch.randperm(half).cuda()
    x2[:half] = x2[rp]
    target[:half] = 0
    target[half:] = 1

    out = discriminator(torch.cat([x1,x2],dim=1))
    bce = bce_criterion(out, Variable(target))

    bce.backward()
    discriminator_optimizer.step()

    pa = out[half:].detach().mean().cpu().numpy()
    na = out[:half].detach().mean().cpu().numpy()

    acc =out[:half].le(0.5).sum() + out[half:].gt(0.5).sum()

    return bce.data.cpu().numpy(), float(acc.data.cpu().numpy())/bs, pa, na

def test(x):
    time_wise_pa = np.zeros(opt.n_future)
    time_wise_na = np.zeros(opt.n_future)
    time_wise_acc = np.zeros(opt.n_future)

    x1 = x[0]
    bs = x1.size(0)
    for i in range(opt.n_past, opt.n_eval):
        x2 = x[i]
        half = int(bs/2)
        rp = torch.randperm(half).cuda()
        x2[:half] = x2[rp]
        out = discriminator(torch.cat([x1,x2],dim=1))
        #bce = bce_criterion(out, Variable(target))
        pa = out[half:].detach().mean().cpu().numpy()
        na = out[:half].detach().mean().cpu().numpy()

        acc =out[:half].le(0.5).sum() + out[half:].gt(0.5).sum()
        #time_wise_bce.append(bce.data.cpu().numpy())
        time_wise_pa[i-opt.n_past] = pa
        time_wise_na[i-opt.n_past] = na
        time_wise_acc[i-opt.n_past] = float(acc.data.cpu().numpy())/bs

    return time_wise_pa, time_wise_na, time_wise_acc




# --------- training loop ------------------------------------
total_iter = 0

widgets = [Percentage(), ' ', Bar('#'), ' ', Timer(), ' ',ETA(), ' ', FileTransferSpeed()]
for epoch in range(opt.niter):
    discriminator.train()
    epoch_bce = 0.0
    epoch_acc = 0.0
    progress = ProgressBar(widgets=widgets, maxval=opt.epoch_size).start()
    for i in range(opt.epoch_size):
        progress.update(i+1)
        x = next(training_batch_generator)
        # -- generate x
        x = generate_sequence(x)

        bce, acc, pa, na = train(x)
        epoch_bce += bce
        epoch_acc += acc
        writer_train.add_scalar('Train/Positive Activation', pa, total_iter)
        writer_train.add_scalar('Train/Negtive Activation', na, total_iter)
        writer_train.add_scalar('Train/bce_loss', bce, total_iter)
        writer_train.add_scalar('Train/categery_acc', acc, total_iter)
        total_iter += 1

    progress.finish()
    utils.clear_progressbar()

    print('[%02d] bce: %.5f | acc: %.3f (%d)' % (epoch, epoch_bce/opt.epoch_size, float(epoch_acc*100)/opt.epoch_size, epoch*opt.epoch_size*opt.batch_size))
    
    with open(os.path.join(opt.log_dir,'discriminator_losses%s.txt'  %(opt.dataset)),mode='a') as f:
        f.write('%0.8f %.4f \n' %(epoch_bce/opt.epoch_size, float(epoch_acc)/opt.epoch_size))
    # save the model
    save_network(discriminator, 'discriminator', 'last', opt.log_dir, opt.gpu_ids)
    update_learning_rate(optimizers, epoch, opt.lr_decay_iters, gamma = 0.1)

    # Testing:
    discriminator.eval()
    tsize = 50
    time_wise_pa = np.zeros((tsize, opt.n_future))
    time_wise_na = np.zeros((tsize, opt.n_future))
    time_wise_acc = np.zeros((tsize, opt.n_future))
    
    print('Testing epoch %d'%(epoch))
    progress_test = ProgressBar(widgets=widgets, maxval=tsize).start()
    for k in range(tsize):
        progress_test.update(k+1)
        x = next(testing_batch_generator)
        x = generate_sequence(x)

        wise_pa, wise_na, wise_acc = test(x)
        #for t in range(opt.n_past, opt.n_eval):
        time_wise_pa[k,:] = wise_pa
        time_wise_na[k,:] = wise_na
        time_wise_acc[k, :] = wise_acc

    writer_test.add_scalar('Test/Positive Activation', time_wise_pa.mean(), epoch)
    writer_test.add_scalar('Test/Negetive Activation', time_wise_na.mean(), epoch)
    writer_test.add_scalar('Test/Average Accuracy', time_wise_acc.mean(), epoch)

    np.save('%s/results/epoch%s_pa'%(opt.log_dir, str(epoch)), time_wise_pa)
    np.save('%s/results/epoch%s_na'%(opt.log_dir, str(epoch)), time_wise_na)
    np.save('%s/results/epoch%s_acc'%(opt.log_dir, str(epoch)), time_wise_acc)

    progress_test.finish()
    utils.clear_progressbar()
    
    torch.save({
        #'discriminator_opt': discriminator,
        'opt': opt},
        '%s/discriminator_opt.pth' % opt.log_dir)

writer_train.export_scalars_to_json(os.path.join(opt.log_dir, 'train_discriminator_scalar.json'))
writer_train.close()
writer_test.export_scalars_to_json(os.path.join(opt.log_dir, 'test_disciriminator_scalar.json'))
writer_test.close()