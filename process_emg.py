import numpy as np
import ezbci_lite as ez
from scipy import signal
import copy
import matplotlib.pyplot as plt
import json

# Step 1: Load the data
fname = 'data/ollie_2ch000_26May22.xdf'
used_channels = {0:3}
raw_EMG = ez.loadxdf(fname)

raw_EMG['eeg_data'] = raw_EMG['eeg_data'][:, 0:3]       # only save channels you use
raw_EMG['channels'] = {f'EMG{i+1}':i for i in range(3)} # update channels with 3 EMG1, EMG2, EMG3

i_s = 0
i_e = 200

plt.plot(raw_EMG['eeg_time'][i_s:i_e], raw_EMG['eeg_data'][i_s:i_e, 0])
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.show()

plt.plot(raw_EMG['eeg_data'][i_s:i_e, 0])
plt.xlabel('Samples')
plt.ylabel('Amplitude')
plt.show()

# Step 2: Filter the data
coeffs = [0.1, 124.]
num_taps = 31
filt_EMG = ez.filt_cont(raw_EMG, coeffs, num_taps, 'bandpass', 'fir', causal=True)

plt.plot(filt_EMG['eeg_data'][i_s:i_e, 0])
plt.xlabel('Samples')
plt.ylabel('Amplitude')
plt.show()

i_s = 200
i_e = 500
plt.plot(filt_EMG['eeg_data'][i_s:i_e, 0], label='filt')
plt.plot(raw_EMG['eeg_data'][i_s:i_e, 0], label='raw')
plt.xlabel('Samples')
plt.ylabel('Amplitude')
plt.legend(loc='best')
plt.show()

# Step 3: Epoch data
epoch_s = -300  # we want each epoch to start 300 ms before the onset
epoch_e = 2300 # and we want them to end 2.3 seconds after onset
O_mod   = 1000 # for "O" stimuli, we'll add 1 second to account for transitions to/from other states

epoch_C = ez.epoch(filt_EMG, 'C', epoch_s, epoch_e)
epoch_L = ez.epoch(filt_EMG, 'L', epoch_s, epoch_e)
epoch_R = ez.epoch(filt_EMG, 'R', epoch_s, epoch_e)
epoch_U = ez.epoch(filt_EMG, 'U', epoch_s, epoch_e)
epoch_D = ez.epoch(filt_EMG, 'D', epoch_s, epoch_e)
epoch_O = ez.epoch(filt_EMG, 'O', epoch_s+O_mod, epoch_e+O_mod)

epoch_C['erp_data'].shape

# Note the dimenions of "erp_data" are: (epochs x time points x channels)
plt.plot(epoch_C['erp_time'], epoch_C['erp_data'][:, :, 0].T)
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.show()

bl_s = -300 # ms
bl_e = -100 # ms

epoch_C = ez.baseline_correct(epoch_C, bl_s, bl_e)
epoch_L = ez.baseline_correct(epoch_L, bl_s, bl_e)
epoch_R = ez.baseline_correct(epoch_R, bl_s, bl_e)
epoch_U = ez.baseline_correct(epoch_U, bl_s, bl_e)
epoch_D = ez.baseline_correct(epoch_D, bl_s, bl_e)
epoch_O = ez.baseline_correct(epoch_O, bl_s+O_mod, bl_e+O_mod)

plt.plot(epoch_C['erp_time'], epoch_C['erp_data'][:, :, 0].T)
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.show()

# Plot all 3 EMG channels on a single plot by scaling y values
plt.plot(epoch_C['erp_time'], epoch_C['erp_data'][:, :, 0].T + 1500); # Channel 1
plt.plot(epoch_C['erp_time'], epoch_C['erp_data'][:, :, 1].T);        # Channel 2
plt.plot(epoch_C['erp_time'], epoch_C['erp_data'][:, :, 2].T - 1500); # Channel 3
plt.xlabel('Time')
plt.ylabel('Relative Amplitude')
plt.show()

# Part 4: Generate averaged figures
# We'll make a quick function to plot all 3 channels on the same plot
def plot_3_chans(data, t, sep, title='', ax=None):
    if ax == None:
        raise Exception('Must provide axis object')
    
    vals = [-sep, 0, sep] # fixed to 3 values for now
    [ax.plot(t, data[:, i] + x) for i, x in enumerate(vals)]
    ax.set_yticks([-sep, 0, sep])
    ax.set_yticklabels(['EMG1', 'EMG2', 'EMG3'])
    ax.set_xlabel('Time (ms)')
    ax.set_title(f'{title}')

# Create averages across epochs
times   = epoch_C['erp_time'] # same for all except epoch_O
times_O = epoch_O['erp_time']

# returns num time points x channels
avg_C = np.mean(epoch_C['erp_data'], 0) 
avg_L = np.mean(epoch_L['erp_data'], 0) 
avg_R = np.mean(epoch_R['erp_data'], 0) 
avg_U = np.mean(epoch_U['erp_data'], 0) 
avg_D = np.mean(epoch_D['erp_data'], 0) 
avg_O = np.mean(epoch_O['erp_data'], 0)

# Let's create a subplot. We will need 6 plots in total so we can make it 2 cols by 3 rows
fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(10, 10), dpi=300)

plot_3_chans(avg_C, times,   500, 'Clenched', ax[0, 0])
plot_3_chans(avg_L, times,   500, 'Left Rotation', ax[0, 1])
plot_3_chans(avg_R, times,   500, 'Right Rotation', ax[1, 0])
plot_3_chans(avg_U, times,   500, 'Upwards Rotation', ax[1, 1])
plot_3_chans(avg_D, times,   500, 'Downwards Rotation', ax[2, 0])
plot_3_chans(avg_O, times_O, 500, 'Open (relaxed)', ax[2, 1])

fig.tight_layout()
plt.show()

# Part 5: Extract data for the ML team
# Function to 2D-ify our 3D epochs
def two_d_ify(data, label):
    obs, ts, chs = data.shape
    l = []
    for o in range(obs):
        for c in range(chs):
            #         label, observation, channel, all signals
            l.append([label, o,           c,       list(data[o, :, c])])

    return l

# Let's turn our 3D epochs into 2D DFs so that the ML team doesn't have to do it
# We'll export to a dict

export_dict = {s:{} for s in ['C', 'R', 'L', 'U', 'D', 'O']}

export_dict['C']['data'] = two_d_ify(epoch_C['erp_data'], 'C')
export_dict['C']['header'] = ['Label', 'Observation', 'Channel', list(times)]

export_dict['R']['data'] = two_d_ify(epoch_R['erp_data'], 'R')
export_dict['R']['header'] = ['Label', 'Observation', 'Channel', list(times)]

export_dict['L']['data'] = two_d_ify(epoch_L['erp_data'], 'L')
export_dict['L']['header'] = ['Label', 'Observation', 'Channel', list(times)]

export_dict['U']['data'] = two_d_ify(epoch_U['erp_data'], 'U')
export_dict['U']['header'] = ['Label', 'Observation', 'Channel', list(times)]

export_dict['D']['data'] = two_d_ify(epoch_D['erp_data'], 'D')
export_dict['D']['header'] = ['Label', 'Observation', 'Channel', list(times)]

export_dict['O']['data'] = two_d_ify(epoch_O['erp_data'], 'O')
export_dict['O']['header'] = ['Label', 'Observation', 'Channel', list(times_O)]

export_dict['sampling_rate'] = raw_EMG['fs']
export_dict['original_file'] = fname

# Export data to a JSON
with open('for_ML_team000.json', 'w', encoding='utf-8') as f:
    json.dump(export_dict, f, ensure_ascii=False, indent=4)
