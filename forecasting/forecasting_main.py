# ADAPTED FROM https://github.com/interpolants/forecasting/blob/main/navier-stokes/main.py

import os 
import numpy as np
import torch
import torch.nn as nn
import math
import argparse
import matplotlib.pyplot as plt 

#import wandb

from forecasting_utils import (
    clip_grad_norm,
    is_type_for_logging,
    DriftModel,
    determine_device,
    get_forecasting_dataloader,
    get_test_dataloader,
)

from forecasting_interpolant import Interpolant

class Trainer :

    def __init__(self, config, load_path = None, sample_only = False):
        self.config = config
        c = config

        if sample_only:
            assert load_path is not None

        self.I = Interpolant(c)

        self.load_path = load_path
        self.device = determine_device()

        if c.dataset == 'latent_traj':
            self.dataloader = get_forecasting_dataloader(c, c.shuffle_batch)

        self.overfit_batch = next(iter(self.dataloader))

        self.model = DriftModel(c).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=c.base_lr)
        self.step = 0

        if self.load_path is not None:
            self.load()

        self.U = torch.distributions.Uniform(low=c.t_min_train, high=c.t_max_train)
        self.print_config()

    def save(self):
        D = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'step': self.step
        }
        new_path = os.path.join(self.config.data_path, 'forecast_ckpts/')
        if not os.path.exists(new_path):
            os.makedirs(new_path)
        path = os.path.join(new_path, 'latest_model.pt')
        torch.save(D, path)
        print("SAVED CKPT AT ", path)

    def load(self):
        D = torch.load(self.load_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(D['model_state_dict'])
        self.optimizer.load_state_dict(D['optimizer_state_dict'])
        self.step = D['step']
        print("Loaded ckpt, step is", self.step)

    def print_config(self):
        c = self.config
        for key in vars(c):
            val = getattr(c, key)
            if is_type_for_logging(val):
                print(key, val)

    def get_time(self, D):
        D['t'] = self.U.sample(sample_shape = (D['N'],)).to(self.device)
        return D
    
    def wide(self, t):
        return t[:, None]
    
    def drift_to_score(self, D):
        z0, zt = D['z0'], D['zt']
        at, bt, adot, bdot, bF = D['at'], D['bt'], D['adot'], D['bdot'], D['bF']
        st, sdot = D['st'], D['sdot']

        numer = (-bt * bF) + (adot * bt * z0) + (bdot * zt) - (bdot * at * z0)
        denom = (sdot * bt - bdot * st) * st * self.wide(D['t'])

        assert not (torch.any(torch.isnan(numer)) or torch.any(torch.isinf(numer)))
        assert not (torch.any(torch.isnan(denom)) or torch.any(torch.isinf(denom)))
        return numer/denom 
    
    @torch.no_grad()
    def EM(self, base=None, cond=None, diffusion_fn=None, return_avg=False):
        c = self.config
        steps = c.EM_sample_steps
        tmin, tmax = c.t_min_sampling, c.t_max_sampling
        ts = torch.linspace(tmin, tmax, steps).type_as(base)
        dt = ts[1] - ts[0]
        ones = torch.ones(base.shape[0]).type_as(base)

        # initial condition
        xt = base

        def step_fn(xt, t):
            D = self.I.interpolant_coefs({'t': t, 'zt': xt, 'z0': base})
            bF = self.model(xt, t, cond=cond)
            D['bF'] = bF
            sigma = self.I.sigma(t)

            if diffusion_fn is not None:
                g = diffusion_fn(t)
                s = self.drift_to_score(D)
                f = bF + .5 * (g.pow(2) - sigma.pow(2)) * s

            else:
                f = bF
                g = sigma

            mu = xt + f * dt 
            xt = mu + g * torch.randn_like(mu) * dt.sqrt()
            return xt, mu
        
        for i, tscalar in enumerate(ts):
            if i == 0 and (diffusion_fn is not None):
                tscalar = ts[1]
            
            if (i+1) % 5000 == 0:
                print("5000 sample steps")
            xt, mu = step_fn(xt, tscalar*ones)

        assert not (torch.any(torch.isnan(mu)) or torch.any(torch.isinf(mu)))
        if return_avg:
            return mu
        else:
            return xt
    
    @torch.no_grad()
    def definitely_sample(self, plot_samples=False):
        c = self.config
        print("SAMPLING: ")

        self.model.eval()

        D = self.prepare_batch(batch = None, for_sampling=True)

        EM_args = {'base': D['z0'], 'cond': D['cond']}

        diffusion_fns = {
            'g_sigma' : None,
            'g_other' : lambda t: c.sigma_coef * self.wide(1-t).pow(4),
        }

        z1 = D['z1']
        z0 = D['z0']
        cond = D['cond']

        plotD = {}

        for k in diffusion_fns.keys():
            print("sampling for diffusion fn", k)
            sample = self.EM(diffusion_fn=diffusion_fns[k], **EM_args)

            all_tensors = torch.cat([z0, sample, z1], dim=-1)

            mse = ((sample-z1)**2).mean(dim=1)
            #print("Average latent MSE: ", mse.mean().item())
            if plot_samples:
                plt.scatter(z0[:, 0].cpu(), z0[:, 1].cpu(), color='blue', label='z0')
                plt.scatter(z1[:, 0].cpu(), z1[:, 1].cpu(), color='green', label='true z1')
                plt.scatter(sample[:, 0].cpu(), sample[:, 1].cpu(), color='red', label='sampled z1')
                plt.legend()
                plt.title("Latent Forecasting")
                plt.show()

            #plotD[k + "(cond, sample, real)"] = wandb.Image(all_tensors)
        
        #if self.config.use_wandb:
        #    #wandb.log(plotD, step=self.step)

    def get_loader(self):
        return self.dataloader
    
    def get_overfit_batch(self):
        return self.overfit_batch
    
    @torch.no_grad()
    def avg_sampling(self, num_steps=25):
        c = self.config
        print("Calculating average sample error!")

        self.model.eval()
        D = self.prepare_batch(batch = None, for_sampling=True)
        EM_args = {'base': D['z0'], 'cond': D['cond']}

        diffusion_fns = {
            'g_sigma' : None,
            'g_other' : lambda t: c.sigma_coef * self.wide(1-t).pow(4),
        }

        z1 = D['z1']
        
        avg_mse = {}
        for k in diffusion_fns.keys():
            print("sampling for diffusion fn", k)
            mse = 0
            for i in range(num_steps):
                sample = self.EM(diffusion_fn=diffusion_fns[k], **EM_args)
                mse += (((sample-z1)**2).mean(dim=1)).mean().item()
            avg_mse[k] = mse/num_steps
        return avg_mse
    
    @torch.no_grad()
    def return_sample(self, diffusion_fn):
        c = self.config
        self.model.eval()
        D = self.prepare_batch(batch = None, for_sampling=True)
        EM_args = {'base': D['z0'], 'cond': D['cond']}

        z0 = D['z0']
        z1 = D['z1']
        sample = self.EM(diffusion_fn=diffusion_fn, **EM_args)

        return z0, sample, z1
            

    def optimizer_step(self):
        norm = clip_grad_norm(self.model, max_norm = self.config.max_grad_norm)

        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.step += 1
        return norm
    
    def loss_norm(self, x):
        return x.pow(2).sum(-1)
    
    def training_step(self, D):
        assert self.model.training
        model_out = self.model(D['zt'], D['t'], cond=D['cond'])
        target = D['drift_target']
        return self.loss_norm(model_out - target).mean()
    
    
    @torch.no_grad()
    def prepare_batch_latent(self, batch=None, for_sampling=False):
        z0, z1 = batch 

        if for_sampling:
            z0 = z0[:self.config.sampling_batch_size]
            z1 = z1[:self.config.sampling_batch_size]

        z0, z1 = z0.to(self.device), z1.to(self.device)

        D = {
            'z0': z0,
            'z1': z1,
            'N': z0.shape[0]
        }

        return D 
        
    def prepare_batch(self, batch=None, for_sampling=False):
        if batch is None or self.config.overfit:
            batch = self.overfit_batch

        if self.config.dataset == 'latent_traj':
            D = self.prepare_batch_latent(batch, for_sampling=for_sampling)
        
        D = self.get_time(D)

        D['cond'] = D['z0']

        D['noise'] = torch.randn_like(D['z0'])

        D = self.I.interpolant_coefs(D)

        D['zt'] = self.I.compute_zt(D)

        D['drift_target'] = self.I.compute_target(D)

        return D
    
    def sample_ckpt(self):
        print("Not training, just sampling checkpoint")
        assert self.config.use_wandb
        self.definitely_sample()
        print("DONE")

    def do_step(self, batch_idx, batch):
        D = self.prepare_batch(batch)
        self.model.train()
        loss = self.training_step(D)
        loss.backward()
        grad_norm = self.optimizer_step()
        
        if self.step % self.config.print_loss_every == 0:
            print(f"Grad step {self.step}. Loss: {loss.item()}")

        if self.step % self.config.save_every == 0:
            print("saving!")
            self.save()

    def fit(self):
        print("starting fit")

        #is_logging = self.config.use_wandb

        #if is_logging:
        #    self.definitely_sample()
        
        print("starting training:")
        while self.step < self.config.max_steps:
            for batch_idx, batch in enumerate(self.dataloader):
                if self.step >= self.config.max_steps:
                    return 
                self.do_step(batch_idx, batch)

    def test(self, diffusion_fn):
        test_dataloader = get_test_dataloader(self.config)
        print("starting test:")
        avg_mse = 0
        z0s = []
        samples = []
        z1s = []
        for batch_idx, batch in enumerate(test_dataloader):
            D = self.prepare_batch(batch, for_sampling=True)
            EM_args = {'base': D['z0'], 'cond': D['cond']}
            z1 = D['z1'] 
            sample = self.EM(diffusion_fn=diffusion_fn, **EM_args)
            samples.append(sample)
            z0s.append(D['z0'])
            z1s.append(D['z1'])
            avg_mse += (((sample-z1)**2).mean(dim=1)).mean().item()
            if batch_idx % 50 == 0:
                print("batch: ", batch_idx, '/', len(test_dataloader))
        avg_mse = avg_mse / len(test_dataloader)
        print("AVG MSE FOR TESTING: ", avg_mse)
        return torch.cat(z0s, dim=0), torch.cat(samples, dim=0), torch.cat(z1s, dim=0)
    
    def chain_sample(self, cond, diffusion_fn, num_steps=50):
        samples = []
        for i in range(num_steps):
            D = {'z0': cond,
                 'N': cond.shape[0]}
            D = self.get_time(D)
            D['cond'] = D['z0'].clone()
            D['noise'] = torch.randn_like(D['z0'])
            D = self.I.interpolant_coefs(D)
            assert(D['cond'] == cond)
            assert(D['cond'] == sample)

            EM_args = {'base': D['z0'], 'cond': D['cond']}

            sample = self.EM(diffusion_fn=diffusion_fn, **EM_args)
            assert(D['cond'] == cond)
            samples.append(sample)

            cond = sample 
        return samples 


        


class Config:
    def __init__(self, dataset, debug, overfit, sigma_coef, beta_fn,
                 latent_dim, hidden_dim, num_layers,
                 data_path, latent_filename, time_lag, stride, random_time_sampling, shuffle_batch):
        self.dataset = dataset
        self.debug = debug
        print("DEBUG IS", self.debug)

        self.sigma_coef = sigma_coef
        self.beta_fn = beta_fn
        self.EM_sample_steps = 500
        self.t_min_sampling = 0.0
        self.t_max_sampling = .999 

        if self.dataset == 'latent_traj':
            self.data_path = data_path
            self.latent_filename = latent_filename
            self.time_lag = time_lag
            self.stride = stride
            self.random_time_sampling = random_time_sampling
            self.shuffle_batch = shuffle_batch
            
        else:
            assert False

        self.num_workers = 4
        self.delta_t = 0.5
        self.use_wandb = False

        self.overfit = overfit
        print(f"OVERFIT MODE (USEFUL FOR DEBUGGING) IS {self.overfit}")

        if self.debug:
            self.EM_sample_steps = 10
            self.sample_every = 10
            self.print_loss_every = 10
            self.save_every = 10000000
        else:
            self.sample_every = 1000
            self.print_loss_every = 1000
            self.save_every = 1000

        # some training hparams
        self.batch_size = 32 
        self.sampling_batch_size = 4
        self.t_min_train = 0.0
        self.t_max_train = 1.0
        self.max_grad_norm = 1.0
        self.base_lr = 2e-4
        self.max_steps = 1_000_000

        # arch
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['latent_traj'], default='latent_traj')
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--latent_filename', type=str, default='test_latents.pt')
    parser.add_argument('--time_lag', type=int, default=1)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--latent_dim', type=int, default=4)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--num_layers', type=int, default=5)
    parser.add_argument('--random_time_sampling', type=bool, default=False)
    parser.add_argument('--shuffle_batch', type=bool, default=True)
    parser.add_argument('--load_path', type=str, default=None)
    parser.add_argument('--sigma_coef', type=float, default=1.0)
    parser.add_argument('--beta_fn', type=str, default='t^2', choices=['t', 't^2'])
    parser.add_argument('--debug', type=int, default=0)
    parser.add_argument('--sample_only', type=int, default=0)
    parser.add_argument('--overfit', type=int, default=0)
    args = parser.parse_args()

    for k in vars(args):
        print(k, getattr(args, k))
    
    if args.dataset == 'latent_traj':
        conf = Config(
            dataset=args.dataset,
            debug = bool(args.debug),
            overfit = bool(args.overfit),
            sigma_coef = args.sigma_coef,
            beta_fn = args.beta_fn,
            latent_dim = args.latent_dim,
            hidden_dim = args.hidden_dim,
            num_layers = args.num_layers,
            data_path = args.data_path,
            latent_filename = args.latent_filename,
            time_lag = args.time_lag,
            stride = args.stride,
            random_time_sampling= args.random_time_sampling,
            shuffle_batch= args.shuffle_batch,
        )
    else:
        assert False 

    trainer = Trainer(
        conf,
        load_path = args.load_path,
        sample_only = bool(args.sample_only)
    )

    if bool(args.sample_only):
        trainer.sample_ckpt()
    else:
        trainer.fit()
    
