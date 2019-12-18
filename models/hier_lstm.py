import torch
import torch.nn as nn
from torch.autograd import Variable

class base_lstm(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, n_layers):
        super(base_lstm, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        #self.batch_size = batch_size
        self.n_layers = n_layers
        self.embed = nn.Linear(input_size, hidden_size)
        self.lstm = nn.ModuleList([nn.LSTMCell(hidden_size, hidden_size) for i in range(self.n_layers)])
        self.output = nn.Sequential(
                nn.Linear(hidden_size, output_size),
                #nn.BatchNorm1d(output_size),
                nn.Tanh())
        self.hidden = self.init_hidden()

    def init_hidden(self, batch_size):
        hidden = []
        for i in range(self.n_layers):
            hidden.append((Variable(torch.zeros(batch_size, self.hidden_size).cuda()),
                           Variable(torch.zeros(batch_size, self.hidden_size).cuda())))
        return hidden

    def forward(self, input, pre_o = None):
        if pre_o is None:
            embedded = self.embed(input)
        else:
            embedded = self.embed(torch.cat([input, pre_o], -1))
        h_in = embedded
        for i in range(self.n_layers):
            self.hidden[i] = self.lstm[i](h_in, self.hidden[i])
            h_in = self.hidden[i][0]

        return self.output(h_in)


class multi_level_predictor(nn.Module):
    def __init__(self, input_size, embedded_size, z_sizes, output_sizes, multi_n_layers, hidden_sizes):
        super(multi_level_predictor, self).__init__()
        self.input_size = input_size
        self.embedded_size = embedded_size
        self.z_sizes = z_sizes
        self.output_sizes = output_sizes
        self.hidden_sizes = hidden_sizes
        self.multi_n_layers = multi_n_layers
        #self.batch_size = batch_size
        self.n_level = len(output_sizes)
        model_list = []
        pre_os = 0
        self.embed = nn.Linear(input_size, embedded_size)
        for os, zs, hs, n_ls in zip(output_sizes, z_sizes, hidden_sizes, multi_n_layers):
            model_list += [base_lstm(embedded_size + zs + pre_os, os, hs, n_ls)]
            pre_os = os
        self.hier_predictor = nn.ModuleList(model_list)
        
    def init_hidden_states(self, batch_size):
        for l in range(self.n_level):
            self.hier_predictor[l].hidden = self.hier_predictor[l].init_hidden(batch_size)

    def forward(self, input, zs):
        embedded_input = self.embed(input.view(-1, self.input_size))
        out_list = []
        out = None
        for l in range(self.n_level):
            out = self.hier_predictor[l](torch.cat([embedded_input, z[l]], -1), out)
            out_list.append(out)
            
        return torch.cat(out_list, -1)

class base_gaussian(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, n_layers):
        super(base_gaussian, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        #self.batch_size = batch_size
        self.embed = nn.Linear(input_size, hidden_size)
        self.lstm = nn.ModuleList([nn.LSTMCell(hidden_size, hidden_size) for i in range(self.n_layers)])
        self.mu_net = nn.Linear(hidden_size, output_size)
        self.logvar_net = nn.Linear(hidden_size, output_size)
        self.hidden = self.init_hidden()

    def init_hidden(self, batch_size):
        hidden = []
        for i in range(self.n_layers):
            hidden.append((Variable(torch.zeros(batch_size, self.hidden_size).cuda()),
                           Variable(torch.zeros(batch_size, self.hidden_size).cuda())))
        return hidden

    def reparameterize(self, mu, logvar):
        logvar = logvar.mul(0.5).exp_()
        eps = Variable(logvar.data.new(logvar.size()).normal_())
        return eps.mul(logvar).add_(mu)

    def forward(self, input, pre_z=None):
        if pre_z is None:
            embedded = self.embed(input)
        else:    
            embedded = self.embed(torch.cat([input, pre_z], -1))
        
        h_in = embedded
        for i in range(self.n_layers):
            self.hidden[i] = self.lstm[i](h_in, self.hidden[i])
            h_in = self.hidden[i][0]
        mu = self.mu_net(h_in)
        logvar = self.logvar_net(h_in)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar


class multi_level_gaussian(nn.Module):
    def __init__(self, input_size, output_sizes, multi_n_layers, hidden_sizes):
        super(multi_level_gaussian, self).__init__()
        self.input_size = input_size
        self.output_sizes = output_sizes
        self.hidden_sizes = hidden_sizes
        self.multi_n_layers = multi_n_layers
        #self.batch_size = batch_size
        self.n_level = len(output_sizes)
        model_list = []
        pre_is = 0
        for os, hs, n_ls in zip(output_sizes, hidden_sizes, multi_n_layers):
            model_list += [base_gaussian(input_size + pre_is, os, hs, n_ls)]
            pre_is = os
        self.hier_models = nn.ModuleList(model_list)
        
    def init_hidden_states(self, batch_size):
        for l in range(self.n_level):
            self.hier_models[l].hidden = self.hier_models[l].init_hidden(batch_size)

    def forward(self, input):
        input = input.view(-1, self.input_size)
        zs = [], mus = [], logvars = []
        z = None
        for l in range(self.n_level):
            z, mu, logvar = self.hier_models[l](input, z)
            zs.append(z)
            mus.append(mu)
            logvars.append(logvar)

        return zs, mus, logvars
            
