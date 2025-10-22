import gvp
from gvp import GVP, GVPConvLayer, LayerNorm
import math
import numpy as np
from torch_geometric.nn import global_mean_pool
import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm
from tqdm import tqdm

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


class TransformerEncoder(torch.nn.Module):
    def __init__(self, latent_dim, max_seq_len, nhead=4, num_layers=2, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.nhead = nhead
        self.num_layers = num_layers

        self.transformer_layer = torch.nn.TransformerEncoderLayer(d_model=latent_dim,
                                                                  nhead=self.nhead,
                                                                  dim_feedforward=self.latent_dim*4,
                                                                  dropout=dropout,
                                                                  activation='gelu',
                                                                  batch_first=True)
        self.trans_encoder = torch.nn.TransformerEncoder(self.transformer_layer, num_layers=self.num_layers)

        # learnable positional encodings! 
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, self.latent_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        causal = torch.triu(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool), diagonal=1)
        self.register_buffer("causal_mask", causal, persistent=False)

    def forward(self, x):
        # data is (B,T,d)
        # positional encoding 
        x = x + self.pos_embed[:, :x.size(1), :]
        attn_mask = self.causal_mask[:x.size(1), :x.size(1)]
        x = self.trans_encoder(x, mask=attn_mask)
        return x 