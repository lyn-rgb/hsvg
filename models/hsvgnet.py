import torch
import torch.nn as nn
from torch.nn import functional as F

import hier_lstm as lstm
import utils

def make_layer(in_channels, scale_factor=1):
    model = [nn.Conv2d(in_channels, in_channels//2, 1),
                nn.BatchNorm2d(in_channels//2),
                nn.LeakyReLU(0.2, inplace=True)]
    if scale_factor > 1:
        model += [nn.MaxPool2d(scale_factor, scale_factor)]

    return nn.Sequential(*model)

class hsvgnet(nn.Module):
    def __init__(self, opts):
        super(hsvgnet, self).__init__()
        # -------------------------------------- model hyper-parameters
        self.ichan = opts.in_channels or 1
        self.isize = opts.in_size or 64
        self.mtype = opts.model_type or 'dcgan'
        self.nlevel = opts.nlevel or 1
        self.zdims = opts.z_dims or [64]
        self.odims = opts.o_dims or [64]
        self.rsizes = opts.rnn_size or [128]
        self.rnnlayers = opts.rnn_nlayers or [1]
        #self.batchsize = opts.batchsize or 8
        self.gdim = opts.g_dim or 128
        self.total_dim = 0
        for z_dim in self.zdims:
            self.total_dim = self.total_dim + z_dim
        self.total_out_dim = 0
        for o_dim in self.odims:
            self.total_out_dim = self.total_out_dim + o_dim
        
        self.rec_criterion = opts.rec_criterion or None
        self.kld_criterion = opts.kld_criterion or None
        # -------------------------------------- construct hsvgnet
        ## Encoder & Decoder
        if self.mtype == 'vgg':
            if self.isize == 128:
                import vgg_128 as backbone
            else:
                import vgg_64 as backbone
        else:
            if self.isize == 128:
                import dcgan_128 as backbone
            else:
                import dcgan_64 as backbone
        
        self.encoder = backbone.encoder(self.gdim, self.ichan)
        self.decoder = backbone.decoder(self.total_out_dim, self.ichan)

        ## multi-level features
        if self.isize == 128:
            self.L1 = make_layer(256, 8)
            self.L2 = make_layer(512, 4)
            self.L3 = make_layer(512, 2)
            self.out_layer = nn.Sequential(nn.Conv2d(640, self.gdim, 4, 1, 0),
                                           #nn.BatchNorm2d(self.gdim),
                                           nn.Tanh())
        else:
            self.L1 = make_layer(128, 8)
            self.L2 = make_layer(256, 4)
            self.L3 = make_layer(512, 2)
            self.out_layer = nn.Sequential(nn.Conv2d(448, self.gdim, 4, 1, 0),
                                           #nn.BatchNorm2d(self.gdim),
                                           nn.Tanh())
        self.hsvg_layer = nn.Linear(self.gdim, self.total_dim)
        ## posterior rnn
        self.posterior = lstm.multi_level_gaussian(self.gdim, self.zdims, self.rnnlayers, self.rsizes)
        ## prior rnn
        self.prior = lstm.multi_level_gaussian(self.gdim, self.zdims, self.rnnlayers, self.rsizes)
        ## predictor rnn
        self.predictor = lstm.multi_level_predictor(self.gdim, self.gdim//8, self.zdims, self.odims, self.rnnlayers, self.rsizes)
        ## skip connect
        self.skips = None
        self.prev_h = None
    
    def reconstruction(self, x):
        h, feats = self.encoding(x)
        zs, mus, logvars = self.posterior(h)
        self.zs_prior, self.mus_prior, self.logvars_prior = self.prior(self.prev_h)
        hs_rec = self.predictor(self.prev_h, zs)
        self.prev_h = h
        return hs_rec, feats, zs, mus, logvars

    def forward(self, x, update_skips=False):
        # reconstruction
        hs_rec, feats, zs, mus, logvars = self.reconstruction(x)
        # decoding
        x_rec = self.decoding(hs_rec)
        ## Losses
        rec = 0.0
        kld = 0.0
        if self.rec_criterion is not None:
            # rec loss
            rec = self.rec_criterion(x_rec, x)
            # kl loss
            posterior_mu = torch.cat(mus, -1)
            posterior_logvar = torch.cat(logvars, -1)
            prior_mu = torch.cat(self.mus_prior, -1)
            prior_logvar = torch.cat(self.logvars_prior, -1)
            kld = self.kld_criterion(posterior_mu, posterior_logvar, prior_mu, prior_logvar)
        
        if update_skips:
            self.skips = feats
        return x_rec, rec, kld
    
    def encoding(self, x):
        _, feats = self.encoder(x)
        f1 = self.L1(feats[-3])
        f2 = self.L2(feats[-2])
        f3 = self.L3(feats[-1])
        h = self.out_layer(torch.cat([f1, f2, f3], dim=1)).view(-1, self.gdim)
        return h, feats
    
    def decoding(self, hs):
        h = torch.cat(hs, -1)
        return self.decoder(h self.skips)

    def inference(self):
        # recurrence
        self.zs_prior, self.mus_prior, self.logvars_prior = self.prior(self.prev_h)
        hs_pred = self.predictor(self.prev_h, self.zs_prior)
        # decoding
        x_pred = self.decoding(hs_pred)
        # encoding
        self.prev_h, _ = self.encoding(x_pred)
        
        return x_pred
    
    def init_rnns(self, batch_size):
        self.posterior.init_hidden_states(batch_size)
        self.prior.init_hidden_states(batch_size)
        self.predictor.init_hidden_states(batch_size)
    
    def init_states(self, x):
        ## processing the first given frame
        ### Initialize the hidden states in lstm models
        bs = x.size(0)
        self.init_rnns(bs)
        ### Encoding
        h, feats = self.encoding(x)
        ### initialize posterior
        zs_init, mus_init, logvars_init = self.posterior(h)
        #self.zs_prior, self.mus_prior, self.logvars_prior = self.prior(h)
        self.prev_h = h
        self.skips = feats

    def eval(self):
        self.posterior.eval()
        self.prior.eval()
        self.predictor.eval()





