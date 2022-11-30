# %%
import numpy as np
import quantities as pq
import neo
import elephant
import viziphant
import matplotlib.pyplot as plt
np.random.seed(4542)

A_sync = 0.02
# the computation of ground truth works only with a shift of 0 
shift = 0

def generate_spike(A_sync=A_sync, shift=shift):
    # Function to generate spike trains of 10 neurons 
    # INPUT : A_sync is the probability to have 10 synchronous spikes
    # The rest of the spikes will be randomly put 
    # OUTPUT : spike trains 
    A = [0]+[1.-A_sync]+[0]*8+[A_sync]
    spiketrains = elephant.spike_train_generation.compound_poisson_process(
                             rate=5*pq.Hz, A=A, shift=shift*pq.ms, t_stop=10*pq.s)
    return spiketrains

def add_noise_test(spiketrains):
    # Function to add neurons that has random spike activities (noise)
    # and detect the pattern
    # INPUT : sipke trains 
    # OUTPUT : patterns detected by SPADE
    for i in range(90):
        spiketrains.append(elephant.spike_train_generation.homogeneous_poisson_process(
            rate=5*pq.Hz, t_stop=10*pq.s))
    patterns = elephant.spade.spade(
                                spiketrains=spiketrains, binsize=1*pq.ms, winlen=1, min_spikes=3,
                                n_surr=100,dither=5*pq.ms,
                                psr_param=[0,0,0],
                                output_format='patterns')['patterns']
    return patterns

def ground_truth_spike_synchronous(spiketrains):
    # Function to find the real position of the pattern
    # Works only for synchronus activity, when ALL neurons 
    # are activated at the same time
    # INPUT : spike trains without the added noise
    dict_spiketrains = {}
    # remove the quantity (s) of the spiketrains array
    for neurons in range(len(spiketrains)):
        a = [float(x) for x in spiketrains[neurons]]
        dict_spiketrains[str(neurons)] = a

    list_keys=list(dict_spiketrains.keys())
    common_spikes = set(dict_spiketrains.get(list_keys[0])).intersection(dict_spiketrains.get(list_keys[1]))
    for key in list_keys[1:-1]:
        num_key=int(key)
        common_spikes = set(common_spikes).intersection(dict_spiketrains.get(str(num_key+1)))

    return common_spikes

def plot_spiketrains(spiketrains): 
    # Function to plot the spike trains we generated
    for i, spiketrain in enumerate(spiketrains):
        t = spiketrain.rescale(pq.s)
        plt.plot(t, i * np.ones_like(t), 'k.', markersize=2)
    plt.axis('tight')
    plt.xlim(0, 10)
    plt.xlabel('Time (s)', fontsize=16)
    plt.ylabel('Spike Train Index', fontsize=16)
    plt.gca().tick_params(axis='both', which='major', labelsize=14)
    plt.grid()
    plt.show()  

def eval_spade(pattern, common_spikes):
    # Function to compare the time of the detection with the the real time of the 
    # pattern 
    # INPUT : - pattern : events detected by spade
    #         - common_spikes : ground truth of the event

    idx_detect = pattern[0].get('windows_ids')
    common_spikes = np.array(list(common_spikes))*1000
    real_idx = common_spikes.astype(int)

    tp = len(list(set(idx_detect).intersection(real_idx)))
    fp = len(idx_detect) - tp
    fn = len(real_idx) - tp

    precision = tp / (tp+fp)
    recall = tp / (tp+fn)

    precision=0
    recall=0
    fscore=0
    if (tp+fp):
        precision = tp / (tp+fp)
    if (tp+fn):
        recall = tp / (tp+fn)
    if (precision+recall):
        fscore = 2*precision*recall / (precision + recall)

    return [precision, recall, fscore]

# %%
if __name__ == '__main__' :
    A_sync = np.logspace(-5,-2,10)
    list_pr = []
    list_rc = []
    list_f = []
    for k in A_sync:
    
        spiketrains = generate_spike(A_sync=k, shift=shift)

        common_spikes = ground_truth_spike_synchronous(spiketrains)

        pattern = add_noise_test(spiketrains)

        [precision, recall, fscore] = eval_spade(pattern, common_spikes)
        print(len(spiketrains))
        list_pr.append(precision)
        list_rc.append(recall)
        list_f.append(fscore)

# %%
