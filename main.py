'''
Author: Qi7
Date: 2022-06-26 23:18:49
LastEditors: aaronli-uga ql61608@uga.edu
LastEditTime: 2022-06-27 00:59:59
Description: 
'''

#%%
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from data_loader import SiameseNetworkDataset
# from torchinfo import summary
# from training import train_loop, eval_loop
# from MQTT_dataloader import MQTTLoader, load_file
# from models import LSTM


###### loading data ######
# normal: 0, scan_A: 1, scan_sU: 2, sparta: 3, mqtt_bruteforce: 4 
data_path = "synthetic_dataset.npy"
with open(data_path, 'rb') as f:
    data = np.load(f)

x = data[:, :data.shape[1]-1]  # data
y = data[:, -1] # label


# %%
siamese_dataset = SiameseNetworkDataset(data)
my_dataloader = DataLoader(siamese_dataset, shuffle=True, batch_size=256)

example_batch = next(iter(my_dataloader))

print(example_batch[2].numpy().reshape(-1))
# %%
# while x in iter(my_dataloader):
#     print(x[2].numpy().reshape(-1))
