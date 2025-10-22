import gvp
from gvp import GVP, GVPConvLayer, LayerNorm
import math
import numpy as np
import torch
import torch.nn as nn


class GraphConvEncode(torch.nn.Module):
    def __init__(self, node_in, node_h, edge_in, edge_h, latent_dim=4,
                 n_layers=3,drop_rate=0.1, node_num=5):
        super(GraphConvEncode, self).__init__()
        self.node_in = node_in
        self.node_h = node_h
        self.edge_in = edge_in
        self.edge_h = edge_h
        self.drop_rate = drop_rate
        self.node_num = node_num
        self.latent_dim = latent_dim
        
    
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
                    for _ in range(n_layers) )
        
        squeeze_layers = [
            nn.Linear(self.node_num*(self.node_h[0] + self.node_h[1]*3),128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.drop_rate),
            nn.Linear(128,32),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.drop_rate),
            nn.Linear(32,self.latent_dim),
        ]

        self.squeeze_layer = nn.Sequential(*squeeze_layers)

        
        unsqueeze_layers = [
            nn.Linear(self.latent_dim, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(p=drop_rate),
            nn.Linear(32, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=drop_rate),
            nn.Linear(128, self.node_num*(self.node_h[0]+self.node_h[1]*3)),
        ]

        self.unsqueeze_layer = nn.Sequential(*unsqueeze_layers)
        
        self.decoder_layers = nn.ModuleList(
                GVPConvLayer(self.node_h, self.edge_h, drop_rate=self.drop_rate)
            for _ in range(n_layers) )
        
        self.W_out = GVP(self.node_h, (3,0), activations=(None,None))
        
    
    def forward(self, h_V, edge_index, h_E):
        h_V = (h_V[0].reshape(h_V[0].shape[0]*h_V[0].shape[1], h_V[0].shape[2]) ,
               h_V[1].reshape(h_V[1].shape[0]*h_V[1].shape[1], h_V[1].shape[2], h_V[1].shape[3]))
        
        h_E = (h_E[0].reshape(h_E[0].shape[0]*h_E[0].shape[1], h_E[0].shape[2]) ,
               h_E[1].reshape(h_E[1].shape[0]*h_E[1].shape[1], h_E[1].shape[2], h_E[1].shape[3]))
        
        h_V = self.W_v(h_V)
        h_E = self.W_e(h_E)
        
        for layer in self.encoder_layers:
            h_V = layer(h_V, edge_index, h_E)

        encoder_embeddings = h_V
        
        flat_s = h_V[0].reshape(h_V[0].shape[0]//self.node_num, -1)

        flat_V = h_V[1].reshape(h_V[1].shape[0]//self.node_num, -1)

        h_V_stack = torch.cat((flat_s, flat_V), dim=1)

        h_V_stack = self.squeeze_layer(h_V_stack)
        
        h_V_small = torch.clone(h_V_stack)
        
        h_V_stack = self.unsqueeze_layer(h_V_stack)
        flat_s = h_V_stack[:,:self.node_num*encoder_embeddings[0].shape[1]]
        flat_V = h_V_stack[:,self.node_num*encoder_embeddings[0].shape[1]:]
        h_V = (flat_s.reshape(encoder_embeddings[0].shape),
               flat_V.reshape(encoder_embeddings[1].shape))
        
        for layer in self.decoder_layers:
            h_V = layer(h_V, edge_index, h_E)
        pred = self.W_out(h_V)
        
        pred = pred.reshape(-1, self.node_num, 3)
        
        return pred, h_V_small
    
    def calc_loss(self, x, output, loss_fun, **kwargs):
        emb, _ = output
        loss = loss_fun(x, emb)
        return loss


def pairwise_cos_sim(x):
    x_norm = torch.nn.functional.normalize(x, p=2, dim=-1)  # --> B,(6/3)
    # multiply row i with row j using transpose
    # element wise product
    pairwise_sim = torch.matmul(x_norm, torch.transpose(x_norm, 0, 1))
    return pairwise_sim

def train(model: GraphConvEncode, device, train_dataloader, optimizer, loss_function, epoch, val_int=0, val_dataloader=None):
    model.train()
    batch_losses = []
    batch_tica_err = []
    for i, batch in enumerate(train_dataloader):
        batch = batch.to(device)
        nodes = (batch.node_s, batch.node_v)
        edges = (batch.edge_s, batch.edge_v)
        GT = batch.x
        tica = batch.tica

        optimizer.zero_grad()
        edge_index = batch.edge_index.permute([1, 0, 2])
        edge_index = edge_index.reshape(2, -1)
        output = model(nodes, edge_index, edges)
        emb, latent = output 

        loss = model.calc_loss(GT, output, loss_function)
        tica_err = ((pairwise_cos_sim(tica) - pairwise_cos_sim(latent)) ** 2).mean()
        
        batch_losses.append(loss.item())
        batch_tica_err.append(tica_err.item())
        
        loss.backward()
        optimizer.step()
    
    if val_dataloader and (epoch + 1) % val_int == 0:
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for i, batch in enumerate(val_dataloader):
                batch = batch.to(device)
                nodes = (batch.node_s, batch.node_v)
                edges = (batch.edge_s, batch.edge_v)
                GT = batch.x
                tica = batch.tica

                edge_index = batch.edge_index.permute([1, 0, 2])
                edge_index = edge_index.reshape(2, -1)

                output = model(nodes, edge_index, edges)
                emb, latent = output

                loss = model.calc_loss(GT, output, loss_function)
                val_loss += loss.item()
        val_loss /= len(val_dataloader)
        print(f"Epoch {epoch+1}, Training Loss: {np.mean(batch_losses): .4f}, Validation Loss: {val_loss:.4f}")


    return np.mean(batch_losses), np.mean(batch_tica_err)

def test(model: GraphConvEncode, device, test_dataloader, loss_function):
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for batch in test_dataloader:
            batch = batch.to(device)
            nodes = (batch.node_s, batch.node_v)
            edges = (batch.edge_s, batch.edge_v)
            tica = batch.tica
            GT = batch.x

            edge_index = batch.edge_index.permute([1, 0, 2])
            edge_index = edge_index.reshape(2,-1)
            output = model(nodes , edge_index, edges)

            loss = model.calc_loss(GT, output, loss_function)
            test_loss += loss.item()
    test_loss /= len(test_dataloader)

    rmsd_angstroms = math.sqrt(test_loss) * 10

    print(f"RMSD on test data {rmsd_angstroms:.4f} Å")
    return




