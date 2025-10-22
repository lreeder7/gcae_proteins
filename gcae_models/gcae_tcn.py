import gvp
from gvp import GVP, GVPConvLayer, LayerNorm
import math
import numpy as np
from torch_geometric.nn import global_mean_pool
import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm
from tqdm import tqdm
from torch.nn.utils.parametrizations import weight_norm

class GVPEncoder(torch.nn.Module):
    def __init__(self, node_in, node_h, edge_in, edge_h, latent_dim=4,
                 n_layers=3, drop_rate=0.1, node_num=5):
        super(GVPEncoder, self).__init__()
        self.node_in = node_in
        self.node_h = node_h
        self.edge_in = edge_in
        self.edge_h = edge_h
        self.drop_rate = drop_rate
        self.latent_dim = latent_dim
        self.node_num = node_num 

        hidden1 = 4*latent_dim
        hidden2 = 2*latent_dim 

        self.W_v = nn.Sequential(
            GVP(node_in, node_h, activations=(None, None)),
            LayerNorm(node_h)
        )

        self.W_e = nn.Sequential(
            GVP(edge_in, edge_h, activations=(None, None)),
            LayerNorm(edge_h)
        )

        self.encoder_layers = nn.ModuleList(
            GVPConvLayer(self.node_h, self.edge_h, drop_rate=self.drop_rate)
            for _ in range(n_layers)
        )

        self.squeeze_layers = nn.Sequential(
            nn.Linear(self.node_num*(self.node_h[0] + self.node_h[1]*3), hidden1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.drop_rate),
            nn.Linear(hidden1,hidden2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.drop_rate),
            nn.Linear(hidden2,self.latent_dim),
        )

    def forward(self, h_V, edge_index, h_E):
        h_V = self.W_v(h_V)
        h_E = self.W_e(h_E)

        for layer in self.encoder_layers:
            h_V = layer(h_V, edge_index, h_E)

        flat_s, flat_v = h_V 
        flat_v = flat_v.reshape(flat_v.size(0), -1)

        h_V_stack = torch.cat([flat_s, flat_v], dim=-1)
        h_V_stack = h_V_stack.reshape(-1, self.node_num*(self.node_h[0] + self.node_h[1] * 3))

        z = self.squeeze_layers(h_V_stack)
        return z 
    

class GVPDecoder(torch.nn.Module):
    def __init__(self, node_in, node_h, edge_in, edge_h, latent_dim=4,
                 n_layers=3, drop_rate=0.1, dense_mode=False, node_num=5):
        super(GVPDecoder, self).__init__()
        self.node_in = node_in
        self.node_h = node_h
        self.edge_in = edge_in
        self.edge_h = edge_h
        self.drop_rate = drop_rate
        self.latent_dim = latent_dim
        self.dense_mode = dense_mode
        self.node_num = node_num 

        hidden1 = latent_dim*2 
        hidden2 = latent_dim*4 

        self.unsqueeze_layers = nn.Sequential(
            nn.Linear(self.latent_dim, hidden1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.drop_rate),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.drop_rate),
            nn.Linear(hidden2, self.node_num * (self.node_h[0] + self.node_h[1]*3)),
        )

        self.decoder_layers = nn.ModuleList([
            GVPConvLayer(node_h, edge_h, drop_rate=self.drop_rate)
            for _ in range(n_layers)
        ])

        self.W_out = GVP(node_h, (3, 0), activations=(None, None))

    def forward(self, z, edge_index, h_E):
        
        h_V_stack = self.unsqueeze_layers(z) # (B, N*(100 + 16*3)) = (B, N*(s + v*3))
        s_dim, v_dim = self.node_h # 100, 16
        total_nodes = z.shape[0] * self.node_num  # B * N 

        flat_s = h_V_stack[:, :self.node_num * s_dim].reshape(total_nodes, s_dim) # WANT TO BE (B*N, s)
        flat_v = h_V_stack[:, self.node_num * s_dim:].reshape(total_nodes, v_dim, 3) # WANT TO BE (B*N, v, 3)

        h_V = (flat_s, flat_v)

        for layer in self.decoder_layers:
            h_V = layer(h_V, edge_index, h_E)
        
        pred = self.W_out(h_V)
        return pred 


def pairwise_cos_sim(x):
    x_norm = torch.nn.functional.normalize(x, p=2, dim=-1)  # --> B,(6/3)
    # multiply row i with row j using transpose
    # element wise product
    pairwise_sim = torch.matmul(x_norm, torch.transpose(x_norm, 0, 1))
    return pairwise_sim



# Apply convolution with causal padding: following https://discuss.pytorch.org/t/causal-convolution/3456/4
class CausalConv1d(nn.Conv1d):
    def __init__(self, input_size, output_size, kernel_size, stride=1, dilation=1):
        padding = (kernel_size - 1) * dilation
        super(CausalConv1d, self).__init__(input_size, output_size, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation)

    def forward(self, x):
        x = super(CausalConv1d, self).forward(x)
        if self.padding[0] != 0:
            return x[:, :, :-self.padding[0]]
        return x

class ResidTempBlock(nn.Module):
    def __init__(self, input_size, channel_size, output_size, kernel_size, stride=1, dilation=1, dropout=0, apply_relu=True):
        super(ResidTempBlock, self).__init__()

        # do causal convolution here
        self.conv1 = weight_norm(CausalConv1d(input_size,
                                                             channel_size,
                                                             kernel_size,
                                                             stride=stride,
                                                             dilation=dilation))
        self.relu1 = nn.ReLU()

        self.dropout1 = nn.Dropout(dropout)

        # do causal convolution here
        self.conv2 = weight_norm(CausalConv1d(channel_size,
                                                             output_size,
                                                             kernel_size,
                                                             stride=stride,
                                                             dilation=dilation))

        self.relu2 = nn.ReLU()

        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.relu1, self.dropout1,
                                     self.conv2, self.relu2, self.dropout2)

        self.reshape = nn.Conv1d(input_size, output_size, 1) if input_size != output_size else None

        self.relu = nn.ReLU()
        self.apply_relu = apply_relu

        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.reshape is not None:
          self.reshape.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        residual = x if self.reshape is None else self.reshape(x)
        if self.apply_relu:
            out = self.relu(out + residual)
        else:
            out = out + residual
        return out


class TCNModel(nn.Module):
    def __init__(self, input_size, channel_size, input_length,
                 kernel_size, stride=1, dropout=0):
        super(TCNModel, self).__init__()

        self.input_length = input_length
        self.input_size = input_size
        num_layers = len(channel_size)

        layers=[]
        for i in range(num_layers):
            apply_relu = i < num_layers-1
            dilation_size = 2**i
            in_channels = input_size if i == 0 else channel_size[i-1]
            out_channels = input_size if i == (num_layers-1) else channel_size[i]

            layers.append(ResidTempBlock(in_channels, channel_size[i-1], out_channels,
                                    kernel_size, stride=stride,
                                    dilation=dilation_size, apply_relu=apply_relu))

        self.tcn = nn.Sequential(*layers)

    def forward(self, x):
        # tcn needs data to be (batch_size, input_size, input_length) so we need to swap last 2 dims
        x = torch.transpose(x, 1,2)

        out = self.tcn(x)

        # swap data back so its the correct shape for loss/ etc.
        out = torch.transpose(out, 1, 2)
        #out = out.view(batch_size, input_length, input_size)
        return out

    # def forward_autoregressive(self, x0, max_len):

    #     all_out = torch.zeros(x0.size(0), max_len+1, x0.size(2)).to(x0.device) # preallocate output
    #     all_out[:,0,:] = x0[:,0,:]
    #     for t in tqdm(range(max_len)):
    #         xt = all_out[:, :t+1, :]
    #         next_output = self.forward(xt)[:,-1,:].unsqueeze(1)
    #         all_out[:,t+1,:] = next_output.squeeze(1)
    #     return all_out[:, 1:, :]