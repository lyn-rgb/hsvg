import torch
import torch.nn as nn
from torch.nn import functional as F

import lstm
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
        self.rsize = opts.rnn_size or 128
        self.postlayers = opts.posterior_rnn_nlayers or 1
        self.priorlayers = opts.prior_rnn_nlayers or 1
        self.batchsize = opts.batchsize or 8
        self.gdim = opts.g_dim or 128
        self.total_dim = 0
        for z_dim in self.zdims:
            self.total_dim = self.total_dim + z_dim
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
        self.decoder = backbone.decoder(self.gdim, self.ichan)

        ## multi-level features
        if self.isize == 128:
            self.L1 = make_layer(256, 8)
            self.L2 = make_layer(512, 4)
            self.L3 = make_layer(512, 2)
            self.out_layer = nn.Sequential(nn.Conv2d(640, self.gdim, 4, 1, 0),
                                           nn.BatchNorm2d(self.gdim),
                                           nn.Tanh())
        else:
            self.L1 = make_layer(128, 8)
            self.L2 = make_layer(256, 4)
            self.L3 = make_layer(512, 2)
            self.out_layer = nn.Sequential(nn.Conv2d(448, self.gdim, 4, 1, 0),
                                           nn.BatchNorm2d(self.gdim),
                                           nn.Tanh())
        self.hsvg_layer = nn.Linear(self.gdim, self.total_dim)
        ## posterior rnn



        ## prior rnn
        self.skips = None

    def forward(self, x, update_skips=False, ts=0):
        if update_skips or self.skips==None:
            _, self.skips = self.encoder(x)
        else:


        return rec_x
    
    def inference(self, x, update_skips=False):


        return pred_x





