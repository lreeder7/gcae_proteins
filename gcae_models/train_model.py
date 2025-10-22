import argparse
import dataset
from gcae import GraphConvEncode, pairwise_cos_sim
import math
import mdtraj as md
import numpy as np
import os
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

torch.set_num_threads(1) 

import warnings
warnings.filterwarnings("ignore", message="Using torch.cross without specifying the dim arg is deprecated")

def get_args():    
    parser = argparse.ArgumentParser()

    # Dataset to use
    parser.add_argument('--data', type=str, choices=['pentapeptide', 'fs_peptide', 'chignolin', 'trp-cage', 'villin'], default='pentapeptide',
                        help='MD Dataset, default=pentapeptide')

    # Number of training epochs
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs, default=50')
    
    # Validation interval
    parser.add_argument('--val_int', type=int, default=5,
                        help='Interval to calculate validation loss, default=5')

    # Model type: restrict to two choices
    parser.add_argument('--model', type=str, choices=['GCAE', 'GCVAE'], default='GCAE',
                        help='Model architecture to use, default=GCAE')

    # Batch size
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for dataloader, default=256')
    
    # Num workers
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for dataloader, default=4')

    # Learning rate
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate, default=1e-3')
    
    # Dropout
    parser.add_argument('--dr', type=float, default=0.0,
                        help='Dropout rate, default=0.0')

    # Number of edges (if your graph is thresholded or KNN-based, etc.)
    parser.add_argument('--num_edges', type=int, default=2,
                        help='Number of edges per node, default=2')

    # Number of layers in the model
    parser.add_argument('--num_layers', type=int, default=3,
                        help='Number of GNNConv layers, default=3')
    
    # Latent dimension
    parser.add_argument('--latent_dim', type=int, default=4,
                        help='Dimension of latent embedding, default=4')
    
    # Combination type
    parser.add_argument('--beta', type=str, default='CosineCycle',
                        help='If using GCVAE, beta determines how to combine loss terms (options: "cycle" or any float as a string which will be the fixed value of beta.), default=cycle')
    
    # Model directory
    parser.add_argument('--model_dir', type=str, default=None,
                        help='Path to save model progress')

    # Pre-trained model
    parser.add_argument('--load', type=str, default=None,
                        help='File name to load in state dict for further training. Do not include path, just file name in format {epoch-xx.pt}. The path is provided in the --model_dir argument.')

    # Add pre-training smoothing
    parser.add_argument('--smooth_width', type=int, default=None,
                        help='Amount to smooth using Butterworth filter: cutoff frequency becomes 2/width, must be integer greater than 2')
    
    # Add latent space regularization
    parser.add_argument('--latent_reg', type=float, default=0.0,
                        help='Add regularization term with given coefficient to loss term. Default=0.0')
    
    # Add temporal smoothing 
    parser.add_argument('--temp_smooth', type=float, default=0.0,
                        help='Add temporal smoothing term with given coefficient to loss function. Default=0.0')
     
    return parser.parse_args()

def determine_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


if __name__ == "__main__":
    args = get_args()
    split = False 
    if args.data == 'pentapeptide':
        data_dir = '/scratch/users/lreeder/gcae_experiments/pentapeptide/'
        pdb, files = dataset.load_pentapeptide(data_dir=data_dir)
        files = dataset.reshape_time_window(files, pdb, data_dir=data_dir, num_files=25, traj_len=5001, num_split=1)
        split = True
    elif args.data == 'fs_peptide':
        data_dir = '/scratch/users/lreeder/gcae_experiments/fs_peptide/'
        pdb, files = dataset.load_fs_peptide(data_dir=data_dir, analyze=False)
        files = dataset.reshape_time_window(files, pdb, os.path.join(data_dir, 'sliced'), 28, 10000, 1)
        split = True 
    elif args.data == 'chignolin':
        data_dir = '/scratch/users/lreeder/gcae_experiments/DES_trajectories/'
        pdb, files = dataset.load_DESRES('chignolin', data_dir, analyze=False)
    elif args.data == 'trp-cage':
        data_dir = '/scratch/users/lreeder/gcae_experiments/DES_trajectories/'
        pdb, files = dataset.load_DESRES('trp-cage', data_dir, analyze=False)
    elif args.data == 'villin':
        data_dir = '/scratch/users/lreeder/gcae_experiments/DES_trajectories/'
        pdb, files = dataset.load_DESRES('villin', data_dir, analyze=False)
    else:
        raise ValueError("We currently do not support datasets other than pentapeptide and fs peptide or the 3 DESRES molecules.")

    if files[0][-4:] == '.xtc':
        traj0 =  md.load_xtc(files[0], top=pdb)
    else:
        traj0 = md.load_dcd(files[0], top=pdb)
    top_k = args.num_edges
    n_layers = args.num_layers
    drop_rate = args.dr
    lr = args.lr
    epochs = args.epochs
    latent_dim = args.latent_dim
    val_int = args.val_int

    model_type = args.model

    batch_size = args.batch_size
    num_workers = args.num_workers

    smooth = args.smooth_width

    lambda_reg = args.latent_reg
    lambda_temp = args.temp_smooth

    if args.model_dir:
        model_dir = args.model_dir
        if model_dir[-1] != '/': model_dir = model_dir +'/'
    else:
        model_dir = f'gcae_experiments/{model_type.lower()}_training/{args.data}/nonormal_{smooth}smoothing_{epochs}ep_{lr}lr_{drop_rate}dr_{n_layers}layers_{top_k}edge_{latent_dim}latentdim_{lambda_reg}reg_{lambda_temp}temp/'


    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
        print(f"Saving model info to {model_dir}")
        
    train_files, val_files, test_files = dataset.train_val_test_split(files, train=0.92, val=0.04, test=0.04)


    file_path = os.path.join(model_dir,'train_val_test_split.csv')

    # if not os.path.exists(file_path):
    df = pd.DataFrame({
          'train_files': pd.Series(train_files),
          'val_files': pd.Series(val_files),
          'test_files': pd.Series(test_files)
    })
    df.to_csv(file_path)
    print(f"Split files up into train-val-test and saved to {file_path}.")
    # else:
    #     df = pd.read_csv(file_path)
    #     train_files = df['train_files'].tolist()
    #     val_files = df['val_files'].tolist()
    #     test_files = df['test_files'].tolist()

    device = determine_device()
    print(f"On device: {device}")

    node_h_dim = (100, 16)
    edge_h_dim = (32, 1)
    node_num = md.load(pdb).topology.n_residues

    train_structures = dataset.generate_structures(train_files, pdb, traj0, split=split, smooth=smooth)
    val_structures = dataset.generate_structures(val_files, pdb, traj0, split=split, smooth=smooth)
    test_structures = dataset.generate_structures(test_files, pdb, traj0, split=split, smooth=smooth)

    train_dataloader, val_dataloader, test_dataloader = dataset.make_dataloaders(train_structures,
                                                                         val_structures,
                                                                         test_structures,
                                                                         top_k,
                                                                         batch_size,
                                                                         num_workers)
    
    if model_type == 'GCAE':
        model = GraphConvEncode((6,3), node_h_dim, (32,1), edge_h_dim,
                                latent_dim=latent_dim,
                                n_layers= n_layers,
                                drop_rate= drop_rate,
                                node_num= node_num,
                                activation=True).to(device)

        
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_function = nn.MSELoss()
    epoch_start = 0

    
    if args.load is not None:
        if os.path.isfile(os.path.join(model_dir,args.load)):
            state_dict = torch.load(os.path.join(model_dir,args.load), map_location=device)  # or 'cuda' if needed
            model.load_state_dict(state_dict)
            model.to(device) 
            epoch_start = int(args.load[6:-3]) + 1
        else:
            raise FileNotFoundError(f"Model file not found at {os.path.join(model_dir,args.load)}")
        
    
    t = epoch_start * len(train_dataloader)

    losses = {}
    val_losses = []

    for epoch in tqdm(range(epoch_start, epochs), desc="Training Progress", leave=True):
        model.train()
        batch_losses = []
        batch_tica_err = []
        
        for i, batch in enumerate(train_dataloader):
            optimizer.zero_grad()

            batch = batch.to(device)
            nodes = (batch.node_s, batch.node_v)
            edges = (batch.edge_s, batch.edge_v)
            GT = batch.x 
            tica = batch.tica

            edge_index = batch.edge_index.permute([1, 0, 2])
            edge_index = edge_index.reshape(2, -1)

            pred, z  = model(nodes, edge_index, edges)
            reg = torch.mean(torch.norm(z, dim=1)**2)

            temp = torch.mean((z[:, 1:] - z[:, :-1])**2)
            

            loss_A = loss_function(GT, pred)
            loss = loss_A + lambda_reg * reg + lambda_temp * temp #1e-6 * reg
            tica_err = ((pairwise_cos_sim(tica) - pairwise_cos_sim(z)) ** 2).mean()

            batch_losses.append(loss.item())
            batch_tica_err.append(tica_err.item())

            loss.backward()
            optimizer.step()
            t += 1 

        train_loss = np.mean(batch_losses)
        losses[epoch] = (train_loss, np.mean(batch_tica_err))
        torch.save(losses, os.path.join(model_dir, 'losses.pt'))


        if (epoch + 1) % val_int == 0:
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
                    

                    pred, z = model(nodes, edge_index, edges)

                    reg = torch.mean(torch.norm(z, dim=1)**2)

                    temp = torch.mean((z[:, 1:] - z[:, :-1])**2)

                    loss_A = loss_function(GT, pred)
                    loss = loss_A + lambda_reg * reg + lambda_temp * temp



                    val_loss += loss.item()
            val_loss /= len(val_dataloader)
            val_losses.append(val_loss)
            print(f"Epoch {epoch+1}/{epochs}, Training Loss: {train_loss: .4f}, Validation Loss: {val_loss:.4f}")
            path = os.path.join(model_dir, f"epoch-{epoch}.pt")
            torch.save(model.state_dict(), path)

    path = os.path.join(model_dir, f"epoch-{epoch}.pt")
    torch.save(model.state_dict(), path)

    if epoch > 16:
        losses_values = list(losses.values())
        loss = [loss[0] for loss in losses_values]
        print("Last few losses: ", loss[-15:])
    print("------------------------------------------------")
    print("EVALUATING:-------------------------------------")

    model.eval()

    test_loss = 0.0
    all_mu = []
    with torch.no_grad():
        for i, batch in enumerate(test_dataloader):

            batch = batch.to(device)
            nodes = (batch.node_s, batch.node_v)
            edges = (batch.edge_s, batch.edge_v)
            GT = batch.x
            tica = batch.tica

            edge_index = batch.edge_index.permute([1, 0, 2])
            edge_index = edge_index.reshape(2, -1)
            pred, z = model(nodes, edge_index, edges)

            all_mu.append(z.cpu())

            reg = torch.mean(torch.norm(z, dim=1)**2)
            temp = torch.mean((z[:, 1:] - z[:, :-1])**2)

            loss_A = loss_function(GT, pred)
            loss = loss_A + lambda_reg * reg + lambda_temp * temp

            test_loss += loss.item()

    all_mu = torch.cat(all_mu, dim=0)
    latent_variances = torch.var(all_mu, dim=0)
    test_loss /= len(test_dataloader)


    rmsd_angstroms = math.sqrt(test_loss) * 10

    print(f"RMSD on test data {rmsd_angstroms:.4f} Å")

    print("Per-latent-dimension variance across dataset:")
    for i, var in enumerate(latent_variances):
        print(f"Dimension {i}: variance = {var.item():.4f}")
    
    torch.save(all_mu, os.path.join(model_dir, "test_latents.pt"))
