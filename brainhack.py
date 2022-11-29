import os

datetag = '2022-11-28'

import torch
torch.set_printoptions(precision=3, linewidth=140, sci_mode=False)

if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
device = torch.device('cpu')


import numpy as np
phi = np.sqrt(5)/2 + 1/2
import matplotlib
import matplotlib.pyplot as plt

import matplotlib
subplotpars = matplotlib.figure.SubplotParams(left=0.125, right=.95, bottom=0.25, top=.975, wspace=0.05, hspace=0.05,)

figpath = '../../2022-11-23_THC-CoSyNe_637e00e2c76207e2838c427d'
figpath = None

def printfig(fig, name, ext='pdf', figpath=figpath, dpi_exp=None, bbox='tight'):
    fig.savefig(os.path.join(figpath, name + '.' + ext), dpi = dpi_exp, bbox_inches=bbox, transparent=True)

# https://docs.python.org/3/library/dataclasses.html?highlight=dataclass#module-dataclasses
from dataclasses import dataclass, asdict, field

@dataclass
class Params:
    datetag: str = datetag
    N_pre: int = 144 # number of presynaptic inputs
    N_PG_time: int = 71 # number of timesteps in PG, must be a odd number for convolutions
    N_PGs: int = 35 # number of polychronous groups
    E_PG: float = 10. # excitability range of PGs
    p_PG: float = .005 # ratio of non-zero coefficients in PGs
    tau_decay: float = .4 # time constant for the decay of the PG enveloppe
    tau_rise: float = .05 # time constant for the rise of the PG enveloppe

    ## Raster plots
    N_time: int =  2**10 # number of timesteps for the raster plot
    N_trials: int = 10 # number of trials
    p_B: float = .001 # prior probability of firing for postsynaptic raster plot
    p_A: float = .001 # prior probability of firing for presynaptic raster plot
    seed: int = 42 # seed

    ## figures
    verbose: bool = False # Displays more verbose output.
    fig_width: float = 12 # width of figure
    phi: float = 1.61803 # beauty is gold
    N_PG_show: float = 5 # number of PG to show in plot_PG

class ABCD:

    def __init__(self, opt):
        self.opt: Params = opt
        self.init()

    def init(self):
        self.logit_p_A = torch.logit(torch.tensor(self.opt.p_A))

        temporal_mod = torch.zeros(self.opt.N_PG_time)
        time = torch.linspace(0, 1, self.opt.N_PG_time)
        temporal_mod = torch.exp(- time / self.opt.tau_decay)
        temporal_mod *= 1 - torch.exp(- time / self.opt.tau_rise)
        # temporal_mod = temporal_mod.flip([0])
        #self.temporal_mod = (torch.eye(self.opt.N_PGs).unsqueeze(2)) * (temporal_mod.unsqueeze(0).unsqueeze(0))
        self.temporal_mod = torch.ones((self.opt.N_pre, self.opt.N_PGs, 1)) * (temporal_mod.unsqueeze(0).unsqueeze(0))

        spike = torch.tensor([1, -.8, -.2])
        self.spike = (torch.eye(self.opt.N_PGs).unsqueeze(2)) * (spike.flip([0]).unsqueeze(0).unsqueeze(0))


    def set_PG(self, seed=None, seed_offset=0):
        if seed is None: seed = self.opt.seed + seed_offset
        torch.manual_seed(seed)
        
        # 1/ define PGs as matrices to be used as kernels
        PG = self.opt.E_PG * torch.randn(self.opt.N_pre, self.opt.N_PGs, self.opt.N_PG_time)
        #threshold = torch.abs(PG).quantile(1-self.opt.p_PG)

        # 2/ zero out everything below the threshold
        # TODO : get analytically
        from scipy.stats import norm
        threshold = self.opt.E_PG * norm.ppf(1-self.opt.p_PG)
        PG *= (PG > threshold)

        # 3/ modulate in time
        PG *= self.temporal_mod

        # 4/ convolve with a spike shape to induce some sort of refractory period
        PG = torch.conv1d(PG, self.spike, padding=self.spike.shape[-1]//2)
        return PG

    def get_b(self, seed=None, seed_offset=1):
        if seed is None: seed = self.opt.seed + seed_offset
        torch.manual_seed(seed)
        # draw causes (PGs) as a matrix of sparse PG activations uniformly in postsynaptic space and time
        # to avoid border effects with the temporal convolution,
        # we set it to zero everywhere
        b_proba = torch.zeros(self.opt.N_trials, self.opt.N_PGs, self.opt.N_time)
        # except outside the borders
        b_proba[:, :, (self.opt.N_PG_time//2):-(self.opt.N_PG_time//2)] = self.opt.p_B
        # b_proba = torch.zeros(self.opt.N_trials, self.opt.N_PGs, self.opt.N_time) * self.opt.p_B
        
        
        return torch.bernoulli(b_proba)

    def plot_raster(self, raster, raster_post=None, PG=None, 
                    i_trial=0, xticks=6, yticks=16, spikelength=.9, 
                    colors=None, figsize=None, subplotpars=subplotpars, 
                    ylabel='address', linewidths=1.0):

        N_neurons = raster.shape[1]
        if colors is None: # blue if nothing assigned
            colors = ['b'] * N_neurons
        else: # give the colors or ...
            if len(colors)==1: # ... paint everything the same color
                colors = colors[0] * N_neurons

        if figsize is None: figsize = (self.opt.fig_width, self.opt.fig_width/self.opt.phi)

        fig, ax = plt.subplots(1, 1, figsize=figsize, subplotpars=subplotpars)
        if raster_post is None:
            for i in range(0, N_neurons):
                ax.eventplot(np.where(raster[i_trial, i, :] == 1.)[0], 
                    colors=colors[i], lineoffsets=1.*i+spikelength/2, 
                    linelengths=spikelength, linewidths=linewidths)
        else:
            for i_PG in range(self.opt.N_PGs):
                b_ = torch.zeros_like(raster_post)
                b_[i_trial, i_PG, :] = raster_post[i_trial, i_PG, :]
                a_ = self.draw_a(b_, PG)
                for i in range(0, self.opt.N_pre):
                    ax.eventplot(np.where(a_[i_trial, i, :] == 1.)[0], colors=colors[i_PG], lineoffsets=1.*i+spikelength/2, 
                    linelengths=spikelength, linewidths=linewidths)
    
        ax.set_ylabel(ylabel)
        ax.set_xlabel('Time (a. u.)')
        ax.set_xlim(0, self.opt.N_time)
        ax.set_ylim(0, N_neurons)

        # ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(yticks))
        ax.set_yticks(np.linspace(0, N_neurons, yticks, endpoint=False)+.5)
        ax.set_yticklabels(np.linspace(1, N_neurons, yticks, endpoint=True).astype(int))
        for side in ['top', 'right']: ax.spines[side].set_visible(False)
        
        ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(4))
        ax.set_xticks(np.linspace(1, self.opt.N_time, xticks, endpoint=True))
        ax.set_xticklabels(np.linspace(1, self.opt.N_time, xticks, endpoint=True).astype(int))

        ax.grid(visible=True, axis='y', linestyle='-', lw=.5)
        ax.grid(visible=True, axis='x', which='both', linestyle='-', lw=.1)

        return fig, ax


    def plot_b(self, b, i_trial=0, xticks=6, yticks=16, spikelength=.9, colors=None, figsize=None, subplotpars=subplotpars):
        b_shifted = torch.roll(b, self.opt.N_PG_time//2, dims=-1)

        fig, ax = self.plot_raster(raster=b_shifted, i_trial=i_trial, xticks=xticks, yticks=yticks, spikelength=spikelength, colors=colors, figsize=figsize, subplotpars=subplotpars, ylabel='@ Motif')
        return fig, ax

    def model_a_logit(self, b, PG):
        """
        defines the evidence of firing for each presynaptic address over time
        input b -> postsynaptic raster plot
            PG -> polychronous groups as spatio temporal kernels
            p_A -> prior proba of firing for the presynaptic addresses
        
        """
        logit_B = torch.conv1d(b*1., PG, padding=self.opt.N_PG_time//2)
        return self.logit_p_A + logit_B

    def model_a(self, b, PG):
        """
        defines the probability of firing for each presynaptic address over time from its evidence
        
        """
        logit_A = self.model_a_logit(b, PG)
        return torch.sigmoid(logit_A)

    def draw_a(self, b, PG, seed=None, seed_offset=2):
        # draws binary events from the probability of firing
        if seed is None: seed = self.opt.seed + seed_offset
        torch.manual_seed(seed)
        # generate the corresponding raster plot
        A_proba = self.model_a(b, PG)
        return torch.bernoulli(A_proba)

    def plot_a(self, a, b=None, PG=None, i_trial=0, xticks=6, yticks=16, spikelength=.9, colors=None, figsize=None, subplotpars=subplotpars):
        fig, ax = self.plot_raster(raster=a, raster_post=b, PG=PG, i_trial=i_trial, xticks=xticks, yticks=yticks, spikelength=spikelength, colors=colors, figsize=figsize, subplotpars=subplotpars, ylabel='@ Neuron')
        return fig, ax

    def inference_with_PGs(self, a, b, PG, max_quant=10000000):
        # infer 
        b_hat = torch.conv_transpose1d(a*1., PG, padding=self.opt.N_PG_time//2)
        # decision
        p_B = b.sum()/(self.opt.N_trials*self.opt.N_PGs*self.opt.N_time) # ça suppose qu'on connait b.sum()...
        if len(b_hat.ravel()) > max_quant:
            ind_quant = torch.randperm(len(b_hat.ravel()))[:max_quant]
            b_threshold = torch.quantile(b_hat.ravel()[ind_quant], 1-env.opt.p_B)
        else:
            b_threshold = torch.quantile(b_hat, 1-p_B)
        b_hat_bin = (b_hat > b_threshold) * 1.
        # b_hat_bin = torch.bernoulli(b_hat)
        return b_hat, b_hat_bin

    def generative_model(self, seed=None, seed_offset=3):
        if seed is None: seed = self.opt.seed + seed_offset
        torch.manual_seed(seed)
        PG, b = self.set_PG(seed=seed), self.get_b(seed=seed+1)
        a = self.draw_a(b, PG, seed=seed+2)
        return a, b, PG

    def test_model(self, PG, PG_true=None, seed=None, seed_offset=4):
        if seed is None: seed = self.opt.seed + seed_offset
        torch.manual_seed(seed)
        if PG_true is None: PG_true = PG
        # define PGs
        # draw causes (PGs)
        b = self.get_b(seed=seed)
        # generate the corresponding raster plot
        a = self.draw_a(b, PG_true, seed=seed+1)
        # infer 
        b_hat, b_hat_bin = self.inference_with_PGs(a, b, PG)
        # count
        accuracy = torch.mean((b_hat_bin == b)*1.)
        TP = torch.mean(b_hat_bin[b==1]*1.)
        TN = 1-torch.mean(b_hat_bin[b==0]*1.)
        return accuracy, TP, TN
        
    def plot_PG(self, PG, cmap='seismic', colors=None, aspect=None, figsize=None, subplotpars=subplotpars, N_PG_show=None):
        if N_PG_show == None: N_PG_show = self.opt.N_PG_show
        if PG.dtype == torch.bool:
            PG_max = 1
            PG_min = 0
            cmap = 'binary'
        else:
            # PG = PG.numpy()
            PG_max = np.abs(PG).max()#.item()
            PG_min = -PG_max

        if figsize is None: figsize = (self.opt.fig_width, self.opt.fig_width/self.opt.phi)

        fig, axs = plt.subplots(1, N_PG_show, figsize=figsize, subplotpars=subplotpars)
        for i_PG in range(N_PG_show):
            ax = axs[i_PG]
            ax.set_axisbelow(True)

            ax.pcolormesh(PG[:, i_PG, :], cmap=cmap, vmin=PG_min, vmax=PG_max)
            #ax.imshow(PG[:, i_PG, :], cmap=cmap, vmin=PG_min, vmax=PG_max, interpolation='none')
            
            ax.set_xlim(0, PG.shape[2])

            ax.set_xlabel('Delay')
            ax.set_title(f'motif #{i_PG+1}', color='k' if colors is None else colors[i_PG])
            if not aspect is None: ax.set_aspect(aspect)

            ax.set_ylim(0, self.opt.N_pre)
            ax.set_yticks(np.arange(0, self.opt.N_pre, 1)+.5)
            if i_PG>0: 
                ax.set_yticklabels([])
            else:
                ax.set_yticklabels(np.arange(0, self.opt.N_pre, 1)+1)

            for side in ['top', 'right']: ax.spines[side].set_visible(False)
            ax.set_xticks([0, self.opt.N_PG_time//2, self.opt.N_PG_time-1])
            ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(self.opt.N_PG_time//4))
            #ax.xaxis.set_minor_locator(AutoMinorLocator(4))
            #ax.set_xticklabels([-(self.opt.N_PG_time//2), 0, self.opt.N_PG_time//2])
            ax.set_xticklabels([0, (self.opt.N_PG_time//2), (self.opt.N_PG_time)])
            # ax.grid(True, axis='y', linestyle='-', lw=1)
            # ax.grid(True, axis='x', which='both', linestyle='-', lw=.1)

        axs[0].set_ylabel('@ Neuron')
        return fig, axs

    def plot_a_histo(self, a,  xticks=1, spikelength=.9, colors=None, figsize=None, subplotpars=subplotpars):
        if figsize is None: figsize = (self.opt.fig_width, self.opt.fig_width/self.opt.phi)

        fig, ax = plt.subplots(1, 1, figsize=figsize, subplotpars=subplotpars)
        ax.step(np.arange(self.opt.N_pre), a.numpy().mean(axis=(0, 2)), label='firing rate')
        ax.hlines(self.opt.p_A, 0, self.opt.N_pre, linestyles='--', color='orange', label='probability of firing for each address')
        ax.set_xlabel('address')
        ax.set_ylim(0)
        ax.legend()
        return fig, ax

    def plot_inference(self, b, b_hat, b_hat_bin, i_trial = 0, t_min = 100, t_max = 150, figsize=None, subplotpars=subplotpars):
        if figsize is None: figsize = (self.opt.fig_width, self.opt.fig_width/self.opt.phi)
        fig, ax = plt.subplots(1, 3, figsize=figsize, subplotpars=subplotpars)
        ax[2].imshow(b_hat_bin[i_trial, :, t_min:t_max])
        ax[0].imshow(b[i_trial, :, t_min:t_max])
        ax[1].imshow(b_hat[i_trial, :, t_min:t_max])
        return fig, ax

    def plot_inference_histo(self, b_hat, nb_bins = 100, figsize=None, subplotpars=subplotpars):
        if figsize is None: figsize = (self.opt.fig_width, self.opt.fig_width/self.opt.phi)

        fig, ax = plt.subplots(1, 1, figsize=figsize, subplotpars=subplotpars)
        ax.hist(b_hat.numpy().ravel(), bins=nb_bins)
        ax.set_ylabel('smarts')
        ax.set_xlabel('value of b_hat')
        ax.set_yscale('log')
        return fig, ax
