import glob
import gvp
import gvp.data
import math
import matplotlib.pyplot as plt
import mdshare
import mdtraj as md
import numpy as np
import os
import pandas as pd 
import random
#from simtk.openmm.app import PDBFile, Simulation, ForceField, CutoffNonPeriodic, HBonds
#from simtk.openmm import VerletIntegrator
#from simtk import unit
from tica import get_linear_TICA
import time
import torch
import torch.nn.functional as F
import torch.utils.data as data
import torch_cluster
import torch_geometric

RES_DICT = {'TRP':'W', 'LEU':'L', 'ALA':'A', 'ARG':'R', 'ASN':'N',
                'ASP':'D', 'CYS':'C', 'GLN':'Q', 'GLU':'E', 'GLY':'G',
                'HIS':'H', 'ILE':'I', 'LYS':'K', 'MET':'M', 'PHE':'F',
                'PRO':'P', 'SER':'S', 'THR':'T', 'TYR':'Y', 'VAL':'V', 'NLE':'O'}


def load_pentapeptide(data_dir='pentapeptide/', analyze=True, plot=False):
    """
    Obtains list of files and pdb for pentapeptide dataset. If files don't exist
    locally, it will download the files from the mdshare website. 
    """
    if os.path.exists(data_dir):
        if data_dir[-1] != '/' : data_dir = data_dir + '/'
        files = glob.glob(data_dir + 'pentapeptide-*-500ns-impl-solv.xtc')
        pdb = data_dir + 'pentapeptide-impl-solv.pdb'
        if os.path.exists(pdb) and len(files) == 25:
            if analyze: analysis(pdb, files, plot)
            return pdb, files
        
    pdb = mdshare.fetch('pentapeptide-impl-solv.pdb', working_directory=data_dir)
    files = mdshare.fetch('pentapeptide-*-500ns-impl-solv.xtc', working_directory=data_dir)
    print(f'We have downloaded {len(files)} trajectories.')

    if analyze: 
        analysis(pdb, files, plot)

    return pdb, files 

def load_fs_peptide(data_dir='fs_peptide/', analyze=True, plot=False):
    """
    Obtains list of files and pdb for fs-peptide dataset, while removing the two 
    unwanted atom selections (ACE and NME). Fs-peptide can be downloaded from 
    https://figshare.com/articles/dataset/Fs_MD_Trajectories/1030363
    """
    if data_dir[-1] != '/': data_dir = data_dir + '/'
    pdb = os.path.join(data_dir, 'fs-peptide.pdb')

    if os.path.exists(os.path.join(data_dir, 'sliced')):
        print(os.path.join(data_dir, 'sliced/'))
        files = glob.glob(os.path.join(data_dir, 'sliced/*.xtc'))
        pdb = os.path.join(data_dir, 'sliced/new_topology.pdb')
        
    else:
        os.makedirs(os.path.join(data_dir, 'sliced'))
        files = glob.glob(os.path.join(data_dir, '*.xtc'))
        files, pdb = slice_out_atoms(['ACE', 'NME'], pdb, files, data_dir)
    print(f'We have downloaded {len(files)} trajectories.')
    

    if analyze:
        analysis(pdb, files, plot)
        
    return pdb, files

def load_DESRES(molecule, data_dir='/scratch/users/lreeder/gcae_experiments/DES_trajectories/', analyze=True, plot=False):
    """
    Obtains list of files and pdb for given molecule in DE Shaw research
    dataset, including chignolin, trp-cage, or villin.  
    Lindorff-Larsen, K., Piana, S., Dror, R. O., & Shaw, D. E. (2011). 
    How fast-folding proteins fold. Science, 334(6055), 517-520.
    """
    if data_dir[-1] != '/': data_dir = data_dir + '/'
    
    if molecule == 'chignolin':
        data_dir = os.path.join(data_dir, 'chignolin/DESRES-Trajectory_CLN025-0-protein/CLN025-0-protein/')
    elif molecule == 'trp-cage':
        data_dir = os.path.join(data_dir, 'trp-cage/DESRES-Trajectory_2JOF-0-protein/2JOF-0-protein/')
    elif molecule == 'villin':
        data_dir = os.path.join(data_dir, 'villin/DESRES-Trajectory_2F4K-0-protein/2F4K-0-protein/')
    else:
        raise ValueError("Molecule must be one of these: chignolin, trp-cage, or villin.")
    
    files = glob.glob(os.path.join(data_dir, '*.dcd'))
    pdb = glob.glob(os.path.join(data_dir, '*.pdb'))[0]
    files.sort()

    first_trj = md.load_dcd(files[0], top=pdb)
    last_trj = md.load_dcd(files[-1], top=pdb)
    if len(first_trj) != len(last_trj):
        files = files[:-1]

    if analyze:
        analysis(pdb, files, plot)

    return pdb, files 


def analysis(pdb, files, plot=False):
    """
    Reports number of atoms, residues, and trajectory lengths for given dataset.
    """
    topology = md.load(pdb).topology
    print(f'Each trajectory contains {topology.n_atoms} atoms and {topology.n_residues} residues.')
    res_seq = []
    print("Residue sequence:   ", end=" ")
    for res in topology.residues:
        print(f'{res.name}', end=' ')
        res_seq.append(RES_DICT[res.name])

    trajectory_lengths = []
    for file in files:
        if file[-4:] == '.xtc':
            traj = md.load_xtc(file, top=pdb)
        else:
            traj = md.load_dcd(file, top=pdb)
        trajectory_lengths.append(len(traj))

    if plot:
        fig, ax = plt.subplots()
        plt.hist(trajectory_lengths)
        plt.title("Trajectory length (number of frames)")
        plt.show()
    
    traj_len = int(np.mean(trajectory_lengths))
    print(f"Trajectory lengths (avg estimate): {traj_len}")
    print(f"Total number of frames (avg estimate): {traj_len*len(files)}")
    return 


def slice_out_atoms(delete_atoms, pdb, files, data_dir):
    """
    Removes 'delete_atoms' atoms from pdb and all files 
    """
    new_files = []
    for i, file in enumerate(files):
        traj = md.load(file, top=pdb)
        exclude_atoms = []
        for atom_name in delete_atoms:
            exclude_atoms.extend([atom.index for atom in traj.topology.atoms if atom.residue.name == atom_name])
        exclude_atoms = set(exclude_atoms)
        include_atoms = [atom.index for atom in traj.topology.atoms if atom.index not in exclude_atoms]

        new_traj = traj.atom_slice(include_atoms)
        new_traj_name = os.path.join(data_dir, f'sliced/traj-{i}.xtc')

        new_traj.save_xtc(new_traj_name)
        new_files.append(new_traj_name)

        if i == 0:
            new_pdb = os.path.join(data_dir, f'sliced/new_topology.pdb')
            new_traj[0].save_pdb(new_pdb)

    return new_files, new_pdb 



def generate_structures(files : list[str], pdb, traj0, split=False, smooth=None, normalize=False):
    """
    Generate structures data from list of files.
    1) Align to traj0
    2) Center coordinates
    3) Grab backbone atoms
    4) Calculate associated tica (for comparisons)
    5) Get coordinates for each frame in each file 
    """

    topology = md.load(pdb).topology
    res_seq = []
    for res in topology.residues:
        res_seq.append(RES_DICT[res.name])


    structures = []
    topology = md.load(pdb).topology
    tica_idx = 0
    tica_idx_dict = {}
    for i, file in enumerate(files):
        if file[-4:] == '.xtc':
            traj = md.load_xtc(file, top=pdb)
        else:
            traj = md.load_dcd(file, top=pdb)
        traj.superpose(traj0,frame=0)
        traj.center_coordinates()

        if smooth is not None:
            traj.smooth(width=smooth, inplace=True)
        
        # grab atoms we want 
        N_atoms = traj.atom_slice(topology.select("protein and name N"))
        C_atoms = traj.atom_slice(topology.select("protein and name C"))
        CA_atoms = traj.atom_slice(topology.select("protein and name CA"))
        O_atoms = traj.atom_slice(topology.select("protein and name O"))
        
        N_coords = N_atoms.xyz
        C_coords = C_atoms.xyz
        CA_coords = CA_atoms.xyz
        O_coords = O_atoms.xyz

        new_coords = np.concatenate([N_coords.reshape(-1, N_atoms.n_atoms*3).T,
                                    C_coords.reshape(-1, C_atoms.n_atoms*3).T,
                                    CA_coords.reshape(-1, CA_atoms.n_atoms*3).T,
                                    O_coords.reshape(-1, O_atoms.n_atoms*3).T])

        D, E = get_linear_TICA(torch.tensor(new_coords), tau=50, d=4)
        tica_data = E@new_coords

        if split:
            traj_file_name = file.split('/')[-1][:-4]
            file_num = int(traj_file_name.split('_')[0][4:])
            chunk_num = int(traj_file_name.split('_')[1][4:])
            frame_idx = chunk_num * len(traj) 
        else:
            file_num = int(file.split('-')[-1][:-4])
        
        for frame in range(0,len(traj)):
            # stack coordinates 
            if split:
                name = f'traj{file_num}_frame{frame_idx}'
                frame_idx += 1 
            else:
                name = f'traj{file_num}_frame{frame}'
            coords = np.stack([N_coords[frame,:,:], C_coords[frame,:,:], CA_coords[frame,:,:], O_coords[frame,:,:]],1)
            
            if normalize:
                flat = coords.reshape(-1, 3)
                rmsd = math.sqrt((flat **2).sum() / flat.shape[0])
                coords = coords / rmsd # Normalize coordinates!


            structures.append({'coords':coords,
                               'name': name,
                               'seq': res_seq,
                               'tica': tica_data[:,frame]})
            ## keep track of location of this (traj,frame) pair in the tica representation
            tica_idx_dict[f'traj{i}_frame{frame}'] = tica_idx
            tica_idx += 1
            
        #print(f"done with file {i}")
        
    print("Structures Length:",len(structures))
    return structures


def reshape_time_window(files, pdb, data_dir='data/', num_files=75, traj_len=1667, num_split = 3):
    """
    Split files into num_split: results in num_files files of length traj_len. 
    NOTE: This will rename file numbers to be from 0 to len(files), which may be
    different than original file naming. 
    """

    path = os.path.join(data_dir, f"split_{num_files}files_{traj_len}len_{num_split}chunks/")
    if not os.path.isdir(path):
        os.makedirs(path)
    else: 
        new_files = sorted([os.path.join(path, f) for f in os.listdir(path) if f.endswith(".xtc")])
        if len(new_files) == num_files:
            return new_files

    files = sorted(files)

    for idx, file in enumerate(files):
        traj = md.load(file, top=pdb)
        assert(traj.n_frames == traj_len*num_split), f"file {idx} has wrong number of frames for splitting."

        for i in range(num_split):
            start = i * traj_len
            end = (i+1) * traj_len

            split = traj[start:end]
            new_traj = f"file{idx}_part{i}.xtc"
            new_path = os.path.join(path, new_traj)
            split.save_xtc(new_path)
    print(f"Split {len(files)} files into {num_files}")
    print("New files created: can be found at", path)
    

    return sorted([os.path.join(path, f) for f in os.listdir(path) if f.endswith('.xtc')]) 


def train_val_test_split(files, train=0.8, val=0.1, test=0.1):
    """
    Split files into train, val, test splits with associated probabilities 
    """
    random.seed(0)
    random.shuffle(files)
    n = len(files)
    train_files = files[0:int(train*n)]
    val_files = files[int(train*n):int((train+val)*n)]
    test_files = files[int((train+val)*n):]

    print("We have a " + str(round(len(train_files)/n,2)*100) + "%-" + 
      str(round(len(val_files)/n,2)*100) + "%-"  + 
      str(round(len(test_files)/n,2)*100) + "% train-val-test split.")
    return sorted(train_files), sorted(val_files), sorted(test_files)

class ProteinDataset(data.Dataset):
    '''
    Adapted from ProteinGraphDataset from B Jing, S Eismann, et al. "Learning 
    from Protein Structure with Geometric Vector Perceptrons"
    https://github.com/drorlab/gvp-pytorch

    '''
    def __init__(self, data_list, device='cpu'):
        super(ProteinDataset, self).__init__()
        
        self.data_list = data_list # considering case where data_list is list of structures (need to run split_preprocess.py first to get this)
        self.device = device
        self.node_counts = 5
        self.top_k = 30
        
        self.letter_to_num = {'C': 4, 'D': 3, 'S': 15, 'Q': 5, 'K': 11, 'I': 9,
                       'P': 14, 'T': 16, 'F': 13, 'A': 0, 'G': 7, 'H': 8,
                       'E': 6, 'L': 10, 'R': 1, 'W': 17, 'V': 19, 
                       'N': 2, 'Y': 18, 'M': 12}
        self.num_to_letter = {v:k for k, v in self.letter_to_num.items()}
        
    def __len__(self): return len(self.data_list)
    
    def __getitem__(self,i):
        return self._featurize_as_graph(self.data_list[i])
    
    def _featurize_as_graph(self, protein):
        name = protein['name']
        #data = super(ProteinDataset, self)._featurize_as_graph(protein)
        with torch.no_grad():
            coords = torch.as_tensor(protein['coords'], device=self.device, dtype=torch.float)
            seq = torch.as_tensor([self.letter_to_num[a] for a in protein['seq']],
                                 device=self.device, dtype=torch.long)
            tica = torch.as_tensor(protein['tica'], device=self.device, dtype=torch.float)
            
            mask = torch.isfinite(coords.sum(dim=(1,2)))
            coords[~mask] = np.inf
            
            X_ca = coords[:, 1] # only consider Carbon alpha atoms here
            edge_index = torch_cluster.knn_graph(X_ca, k=self.top_k)
            
            E_vectors = X_ca[edge_index[0]] - X_ca[edge_index[1]]
            
            orientations = self._orientations(X_ca)
            #print("ORIENTATIONS: ", orientations.shape)
            sidechains = self._sidechains(coords)
            #"print("SIDECHAINS: ", sidechains.shape)
            
            nodes = torch.cat([orientations, sidechains.unsqueeze(-2)], dim=-2)
            edges = _normalize(E_vectors).unsqueeze(-2)
            
            nodes, edges = map(torch.nan_to_num, (nodes, edges))
            
            data = torch_geometric.data.Data(x=X_ca, seq=seq, name=name, tica=tica,
                                             nodes=nodes, edges=edges,
                                            edge_index = edge_index, mask=mask)
            
            return data
        
    ### FOLLOWING IS TAKEN FROM gvp-pytorch/gvp/data.py
        
    def _orientations(self, X):
        forward = _normalize(X[1:] - X[:-1])
        backward = _normalize(X[:-1] - X[1:])
        forward = F.pad(forward, [0, 0, 0, 1])
        backward = F.pad(backward, [0, 0, 1, 0])
        return torch.cat([forward.unsqueeze(-2), backward.unsqueeze(-2)], -2)

    def _sidechains(self, X):
        n, origin, c = X[:, 0], X[:, 1], X[:, 2]
        c, n = _normalize(c - origin), _normalize(n - origin)
        bisector = _normalize(c + n)
        perp = _normalize(torch.cross(c, n))
        vec = -bisector * math.sqrt(1 / 3) - perp * math.sqrt(2 / 3)
        return vec

class LigandDataset(gvp.data.ProteinGraphDataset):
    def __init__(self, data_list,
                 num_positional_embeddings=16,
                 top_k = 30, num_rbf=16, device="cpu",
                 node_counts=5):
        data.Dataset.__init__(self)

        self.data_list = data_list
        self.top_k = top_k
        self.num_rbf= num_rbf
        self.device = device
        self.num_positional_embeddings=num_positional_embeddings
        self.node_counts = node_counts
        self.letter_to_num = {'C': 4, 'D': 3, 'S': 15, 'Q': 5, 'K': 11, 'I': 9,
                       'P': 14, 'T': 16, 'F': 13, 'A': 0, 'G': 7, 'H': 8,
                       'E': 6, 'L': 10, 'R': 1, 'W': 17, 'V': 19, 
                       'N': 2, 'Y': 18, 'M': 12}
        self.num_to_letter = {v:k for k, v in self.letter_to_num.items()}

    def __getitem__(self, i): 
        return self._featurize_as_graph(self.data_list[i])

    def _featurize_as_graph(self, protein):
        # overwrite using from_dict and to_dict (on data)
        # dict.update to add the state information (as a tensor)
        data = super(LigandDataset, self)._featurize_as_graph(protein)
        with torch.no_grad():
            tica = torch.as_tensor(protein['tica'], device=self.device, dtype=torch.float) #file = protein['file'] # torch.as_tensor(protein['file'], device=self.device, dtype=torch.int)
        data_dict = data.to_dict()
        data_dict.update({'tica':tica})
        new_data = torch_geometric.data.Data.from_dict(data_dict)
        return new_data
        

def _normalize(tensor, dim=-1):
    '''
    Normalizes a `torch.Tensor` along dimension `dim` without `nan`s.
    '''
    return torch.nan_to_num(
        torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True)))

class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, trajectory_lengths, sequence_length=512, stride=256, include_partial=False):
        self.sequence_length = sequence_length
        self.stride = stride
        self.sequence_indices = []
        self.global_starts = []
        self.include_partial = include_partial

        start = 0
        for length in trajectory_lengths:
            for i in range(0, length - sequence_length + 1, stride):
                global_start = start + i 
                self.sequence_indices.append([global_start, global_start + sequence_length])
                self.global_starts.append(global_start)
            
            remainder = length % stride
            if self.include_partial and remainder > 0 and remainder < sequence_length:
                last_start = start + length - remainder
                last_end = start + length
                self.sequence_indices.append([last_start, last_end])
                self.global_starts.append(last_start)
            start += length 
    
    def __len__(self):
        return len(self.sequence_indices)
    
    def __getitem__(self, idx):
        return self.sequence_indices[idx]
    
    def get_global_start(self, idx):
        return self.global_starts[idx]


def make_dataloaders(train_structures, val_structures, test_structures, top_k, batch_size, num_workers):
    train_dataset = LigandDataset(train_structures, top_k=top_k)#, device=device)
    val_dataset = LigandDataset(val_structures, top_k=top_k)#, device=device)
    test_dataset = LigandDataset(test_structures, top_k=top_k)#, device=device)

    train_dataloader = torch_geometric.loader.DenseDataLoader(train_dataset, 
                                                       batch_size=batch_size, 
                                                       shuffle=False, 
                                                       num_workers=num_workers)
    val_dataloader = torch_geometric.loader.DenseDataLoader(val_dataset, 
                                                     batch_size=batch_size, 
                                                     shuffle=False, 
                                                     num_workers=num_workers)
    test_dataloader = torch_geometric.loader.DenseDataLoader(test_dataset, 
                                                      batch_size=batch_size, 
                                                      shuffle=False, 
                                                      num_workers=num_workers)
    return train_dataloader, val_dataloader, test_dataloader