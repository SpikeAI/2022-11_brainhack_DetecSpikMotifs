# %%
import numpy as np
import quantities as pq
import neo
import elephant
import viziphant
import matplotlib.pyplot as plt
from brainhack import make_spiketrains_motif, cospattern, linear
from accuracy import eval_spade

# %%
## generate a spiking motif with make_spiketrains_motif function
nb_neuron = 10
noise_density = 5e-3
simulation_time = 100 # in ms
motif_temporal_amplitude = 5
nb_stim = 3
discard_neurons = 1
function = linear #tune here the type of function that you want to represent

t_true = [motif_temporal_amplitude + int(np.random.rand()*(simulation_time-3*motif_temporal_amplitude)) for i in range(nb_stim)]
spike_trains = make_spiketrains_motif(nb_neuron, noise_density, simulation_time, motif_temporal_amplitude, t_true, function= function, discard_spikes = discard_neurons)

# %%
plt.figure(figsize=(15, 7))
plt.eventplot([spike_trains[i].times for i in range(len(spike_trains))], linelengths=0.75, color='black')
plt.xlabel('Time (ms)', fontsize=16)
plt.ylabel('Neuron')
plt.title("Raster plot");


# %%
patterns = elephant.spade.spade(
    spiketrains=spike_trains, binsize=1*pq.ms, winlen=motif_temporal_amplitude, min_spikes=5,
    n_surr=100,dither=0*pq.ms,
    psr_param=[0,0,0],
    output_format='patterns')['patterns']

# %% 
viziphant.patterns.plot_patterns(spike_trains, patterns)

# %%
[precision, recall, fscore] = eval_spade(patterns, t_true)
print(precision)
print(recall)
print(fscore)
# %%
