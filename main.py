'''
Author: Qi7
Date: 2022-06-26 23:18:49
LastEditors: aaronli-uga ql61608@uga.edu
LastEditTime: 2022-06-27 09:00:31
Description: 
'''

#%%
import enum
import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import optim
from torch.utils.data import DataLoader
from data_loader import SiameseNetworkDataset
from models import LSTM
from customized_criterion import ContrastiveLoss
from torchinfo import summary
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
batch_size = 64
siamese_dataset = SiameseNetworkDataset(data)
my_dataloader = DataLoader(siamese_dataset, shuffle=True, batch_size=batch_size)


# example_batch = next(iter(my_dataloader))
# print(example_batch[2].numpy().reshape(-1))
# %%

###### start training ######
print('===============================')
print('start training......')

Lr = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTM(n_inputs=x.shape[1], n_hidden=256, n_layers=1)
model.to(device)
summary(model)
criterion = ContrastiveLoss()
optimizer = optim.Adam(model.parameters(), lr = Lr)

# %%
counter = []
loss_history = [] 
iteration_number= 0

for epoch in range(100):
    
    # Iterate over batches
    for i, (s1, s2, label) in enumerate(my_dataloader, 0):

        # Send the signals to devce(cpu or cuda)

        s1, s2, label = s1.float().to(device), s2.float().to(device), label.float().to(device)

        optimizer.zero_grad()

        output1, output2 = model(s1.view(s1.shape[0], 1, -1), s2.view(s2.shape[0], 1, -1))

        loss_contrastive = criterion(output1, output2, label)

        loss_contrastive.backward()

        optimizer.step()

        if i % 10 == 0:
            print(f"Epoch number {epoch}\n Current loss {loss_contrastive.item()}\n")
            iteration_number += 10

            counter.append(iteration_number)
            loss_history.append(loss_contrastive.item())

plt.plot(counter, loss_history)
plt.show()
# %%
