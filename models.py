'''
Author: Qi7
Date: 2022-06-27 07:17:27
LastEditors: aaronli-uga ql61608@uga.edu
LastEditTime: 2022-06-27 08:58:52
Description: 
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

# classical LSTM model
class LSTM(nn.Module):

    def __init__(self, n_inputs, n_hidden, n_layers):
        super(LSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size = n_inputs,
            hidden_size = n_hidden,
            num_layers = n_layers,
            batch_first = True,
        )

        self.drop = nn.Dropout(p=0.5)
        self.fc = nn.Linear(n_hidden, 5)
        self.relu = nn.ReLU()
        self.hidden_size = n_hidden

        # Don't have to use sotfmax if use crossentropy loss
        # self.softmax = nn.LogSoftmax()
    
    def forward_once(self, x):
        # This function will be called for both signals
        # Its output is used to determine the similiarity
        output, (hn, cn)= self.lstm(x)
        hn = hn.view(-1, self.hidden_size)
        output = self.relu(hn)
        output = self.fc(output)
        return output

    def forward(self, input1, input2):
        # In this function we pass in both signals and obtain both vectors which are returned
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        
        return output1, output2