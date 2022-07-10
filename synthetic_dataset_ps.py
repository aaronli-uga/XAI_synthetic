'''
Author: Qi7
Date: 2022-07-05 16:06:18
LastEditors: aaronli-uga ql61608@uga.edu
LastEditTime: 2022-07-09 22:04:31
Description: Generate synthetic data with different phase shift and amplitude
'''
#%%
import numpy as np
import matplotlib.pyplot as plt
import random

plt.style.use('seaborn-poster')

#%%
# Generate raw waveform signals for different frequencies
# sampling rate
sr = 1000.0

# sampling interval
ts = 1.0 / sr
t = np.arange(0, 1, ts)
ps = 0 #phase shift

t *= 0.1

# frequency of the signal
freq_1 = 80
freq_2 = 130
freq_3 = 180
freq_4 = 230
freq_5 = 280
freq_6 = 330

y_1 = np.sin(1 * np.pi * freq_1 * t + ps)
y_2 = np.sin(1 * np.pi * freq_2 * t)
y_3 = np.sin(1 * np.pi * freq_3 * t)
y_4 = np.sin(1 * np.pi * freq_4 * t)
y_5 = np.sin(1 * np.pi * freq_5 * t)
y_6 = np.sin(1 * np.pi * freq_6 * t)

fig, axs = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle('Raw waveform with different frequencies')
axs[0, 0].plot(t, y_1)
axs[0, 0].set_title('Frequency y_1')
axs[0, 1].plot(t, y_2)
axs[0, 1].set_title('Frequency y_2')
axs[0, 2].plot(t, y_3)
axs[0, 2].set_title('Frequency y_3')
axs[1, 0].plot(t, y_4)
axs[1, 0].set_title('Frequency y_4')
axs[1, 1].plot(t, y_5)
axs[1, 1].set_title('Frequency y_5')
axs[1, 2].plot(t, y_6)
axs[1, 2].set_title('Frequency y_6')

for ax in axs.flat:
    ax.set(xlabel='Time (s)', ylabel='amplitude')

# Hide x labels and tick labels for all but bottom plot.
for ax in fig.get_axes():
    ax.label_outer()

# %%
noise = np.random.normal(0, 0.1, len(t))
y_1 = y_1 + noise
noise = np.random.normal(0, 0.1, len(t))
y_2 = y_2 + noise
noise = np.random.normal(0, 0.1, len(t))
y_3 = y_3 + noise
noise = np.random.normal(0, 0.1, len(t))
y_4 = y_4 + noise
noise = np.random.normal(0, 0.1, len(t))
y_5 = y_5 + noise
noise = np.random.normal(0, 0.1, len(t))
y_6 = y_6 + noise

fig, axs = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle('Raw waveform with different frequencies add noise')
axs[0, 0].plot(t, y_1)
axs[0, 0].set_title('Frequency y_1')
axs[0, 1].plot(t, y_2)
axs[0, 1].set_title('Frequency y_2')
axs[0, 2].plot(t, y_3)
axs[0, 2].set_title('Frequency y_3')
axs[1, 0].plot(t, y_4)
axs[1, 0].set_title('Frequency y_4')
axs[1, 1].plot(t, y_5)
axs[1, 1].set_title('Frequency y_5')
axs[1, 2].plot(t, y_6)
axs[1, 2].set_title('Frequency y_6')

for ax in axs.flat:
    ax.set(xlabel='Time (s)', ylabel='amplitude')

# Hide x labels and tick labels for all but bottom plot.
for ax in fig.get_axes():
    ax.label_outer()
# %% synthetic data make
iteration = 800
s1, s2, s3, s4, s5 = [], [], [], [], []
for i in range(iteration):
    left_ps = random.uniform(0, 2*np.pi)
    left_amp = random.uniform(0, 4)
    right_ps = random.uniform(0, 2*np.pi)
    right_amp = random.uniform(0, 4)
    noise = np.random.normal(0, 0.2, len(t))
    s1.append(
        left_amp*np.sin(1 * np.pi * freq_1 * t + left_ps) + 
        right_amp*np.sin(1 * np.pi * freq_2 * t + right_ps) +
        noise
    )

    left_ps = random.uniform(0, 2*np.pi)
    left_amp = random.uniform(0, 4)
    right_ps = random.uniform(0, 2*np.pi)
    right_amp = random.uniform(0, 4)
    noise = np.random.normal(0, 0.2, len(t))
    s2.append(
        left_amp*np.sin(1 * np.pi * freq_3 * t + left_ps) + 
        right_amp*np.sin(1 * np.pi * freq_4 * t + right_ps) +
        noise
    )

    left_ps = random.uniform(0, 2*np.pi)
    left_amp = random.uniform(0, 4)
    right_ps = random.uniform(0, 2*np.pi)
    right_amp = random.uniform(0, 4)
    noise = np.random.normal(0, 0.2, len(t))
    s3.append(
        left_amp*np.sin(1 * np.pi * freq_5 * t + left_ps) + 
        right_amp*np.sin(1 * np.pi * freq_6 * t + right_ps) +
        noise
    )

    left_ps = random.uniform(0, 2*np.pi)
    left_amp = random.uniform(0, 4)
    right_ps = random.uniform(0, 2*np.pi)
    right_amp = random.uniform(0, 4)
    third_ps = random.uniform(0, 2*np.pi)
    third_amp = random.uniform(0, 4)
    noise = np.random.normal(0, 0.2, len(t))
    s4.append(
        left_amp*np.sin(1 * np.pi * freq_1 * t + left_ps) + 
        right_amp*np.sin(1 * np.pi * freq_2 * t + right_ps) +
        third_amp*np.sin(1 * np.pi * freq_3 * t + third_ps) +
        noise
    )

    left_ps = random.uniform(0, 2*np.pi)
    left_amp = random.uniform(0, 4)
    right_ps = random.uniform(0, 2*np.pi)
    right_amp = random.uniform(0, 4)
    third_ps = random.uniform(0, 2*np.pi)
    third_amp = random.uniform(0, 4)
    noise = np.random.normal(0, 0.2, len(t))
    s5.append(
        left_amp*np.sin(1 * np.pi * freq_4 * t + left_ps) + 
        right_amp*np.sin(1 * np.pi * freq_5 * t + right_ps) +
        third_amp*np.sin(1 * np.pi * freq_6 * t + third_ps) +
        noise
    )


i = random.randint(0, iteration)
fig, axs = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle('Randomly plot the synthetic data')
axs[0, 0].plot(t, s1[i])
axs[0, 0].set_title('Frequency s1')
axs[0, 1].plot(t, s2[i])
axs[0, 1].set_title('Frequency s2')
axs[0, 2].plot(t, s3[i])
axs[0, 2].set_title('Frequency s3')
axs[1, 0].plot(t, s4[i])
axs[1, 0].set_title('Frequency s4')
axs[1, 1].plot(t, s5[i])
axs[1, 1].set_title('Frequency s5')

for ax in axs.flat:
    ax.set(xlabel='Time (s)', ylabel='amplitude')

# Hide x labels and tick labels for all but bottom plot.
for ax in fig.get_axes():
    ax.label_outer()

# %%
s1 = np.array(s1)
s2 = np.array(s2)
s3 = np.array(s3)
s4 = np.array(s4)
s5 = np.array(s5)

npy_data = np.concatenate((s1, s2), axis=0)
npy_data = np.concatenate((npy_data, s3), axis=0)
npy_data = np.concatenate((npy_data, s4), axis=0)
npy_data = np.concatenate((npy_data, s5), axis=0)


num_of_class = npy_data.shape[0] // 5
Y = [0] * num_of_class
Y += [1] * num_of_class
Y += [2] * num_of_class
Y += [3] * num_of_class
Y += [4] * num_of_class
Y = np.array(Y)
Y = Y.reshape((-1,1))

npy_data = np.concatenate((npy_data, Y), axis = 1)

# save the dataset, the last colomn is the label y.
with open('synthetic_dataset_ps.npy', 'wb') as f:
    np.save(f, npy_data)