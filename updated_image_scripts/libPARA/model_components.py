#!/usr/bin/python

"""
Version 4/14/2020

modules for CNN, RNN, LSTM, data loader for segmented cells out of zip file.

"""


import numpy as np
import scipy

import matplotlib.pyplot as plt

import torch
from torch.utils.data import Sampler, Dataset, DataLoader, ConcatDataset
from torch.autograd import Variable
from torchvision import transforms, utils
import timeit

import os
import sys
import re

class convNN(torch.nn.Module):

    def __init__(self):
        super(convNN, self).__init__()

        # img dim = 72

        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(32),
            # torch.nn.MaxPool2d(2)
#             torch.nn.MaxPool2d(kernel_size = 2, stride = 2)
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(32),
            torch.nn.MaxPool2d(2)
#             torch.nn.MaxPool2d(kernel_size = 2, stride = 2)
        )

        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(64),
            # torch.nn.MaxPool2d(2)
#             torch.nn.MaxPool2d(kernel_size = 2, stride = 2)
        )

        self.layer4 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(64),
            torch.nn.MaxPool2d(2)
#             torch.nn.MaxPool2d(kernel_size = 2, stride = 2)
        )

        self.layer5 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(128),
            # torch.nn.MaxPool2d(2)
#             torch.nn.MaxPool2d(kernel_size = 2, stride = 2)
        )

        self.layer6 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(128),
            torch.nn.MaxPool2d(2),
            torch.nn.Dropout(p=0.25)
#             torch.nn.MaxPool2d(kernel_size = 2, stride = 2)
        )
        # self.layer3 = torch.nn.Sequential(
        #     torch.nn.Conv2d(16, 8, kernel_size=7, padding=2),
        #     torch.nn.BatchNorm2d(8),
        #     torch.nn.ReLU(),
        #     torch.nn.MaxPool2d(2)
        #  )

        # self.drop_out = torch.nn.Dropout(p=0.25)
    
        self.fc1 = torch.nn.Linear(18432, 512)
        torch.nn.init.xavier_normal_(self.fc1.weight)
        self.fc2 = torch.nn.Linear(512,64)
        torch.nn.init.xavier_normal_(self.fc2.weight)
        self.fc3 = torch.nn.Linear(64,3)
        torch.nn.init.xavier_normal_(self.fc3.weight)
        
    def forward(self, x):
        # print('wrong forward')
        out = self.layer1(x)
        # print(out.shape)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        # print(out.shape)
        # out = self.layer3(out)
        out = out.view(out.size(0), -1) # flattens out for all except first dimension ( equiv to np. reshape) for fully connected layer
#         print(out.shape)
        # out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)

        return out

class convOnly(torch.nn.Module):

    def __init__(self):
        super(convOnly, self).__init__()

        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(32),
            # torch.nn.MaxPool2d(2)
#             torch.nn.MaxPool2d(kernel_size = 2, stride = 2)
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(32),
            torch.nn.MaxPool2d(2)
#             torch.nn.MaxPool2d(kernel_size = 2, stride = 2)
        )

        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(64),
            # torch.nn.MaxPool2d(2)
#             torch.nn.MaxPool2d(kernel_size = 2, stride = 2)
        )

        self.layer4 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(64),
            torch.nn.MaxPool2d(2)
#             torch.nn.MaxPool2d(kernel_size = 2, stride = 2)
        )

        self.layer5 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(128),
            # torch.nn.MaxPool2d(2)
#             torch.nn.MaxPool2d(kernel_size = 2, stride = 2)
        )

        self.layer6 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(128),
            torch.nn.MaxPool2d(2)
#             torch.nn.MaxPool2d(kernel_size = 2, stride = 2)
        )
        # self.layer3 = torch.nn.Sequential(
        #     torch.nn.Conv2d(16, 8, kernel_size=7, padding=2),
        #     torch.nn.BatchNorm2d(8),
        #     torch.nn.ReLU(),
        #     torch.nn.MaxPool2d(2)
        #  )

        # self.drop_out = torch.nn.Dropout(p=0.3)
    
        self.fc1 = torch.nn.Linear(10368, 512)
        torch.nn.init.xavier_normal_(self.fc1.weight)
        self.fc2 = torch.nn.Linear(512,64)
        torch.nn.init.xavier_normal_(self.fc2.weight)
        
    def forward(self, x):
        # print('wrong forward')
        out = self.layer1(x)
        # print(out.shape)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        # print(out.shape)
        # out = self.layer3(out)
        out = out.view(out.size(0), -1) # flattens out for all except first dimension ( equiv to np. reshape) for fully connected layer
#         print(out.shape)
        # out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)

        return out

class LSTM(torch.nn.Module):
    # Initialize the class
    def __init__(self, x_dim, y_dim, lags, hidden_dim):
        super(LSTM, self).__init__()
        
        # Check if there is a GPU available
        if torch.cuda.is_available() == True:
            self.dtype = torch.cuda.FloatTensor
        else:
            self.dtype = torch.FloatTensor
            
        # X has the form lags x data x dim
        # Y has the form data x dim
                 
        self.X_dim = x_dim
        self.Y_dim = y_dim
        self.hidden_dim = hidden_dim
        self.lags = lags

        self.fc1 = torch.nn.Linear(self.Y_dim, self.Y_dim)
        self.fc2 = torch.nn.Linear(self.Y_dim, 3)
        # Initialize network weights and biases        
        self.W_f, self.W_i, self.W_C, self.W_o, self.U_f, self.U_i, self.U_C, self.U_o, self.b_i, self.b_f, self.b_C, self.b_o, self.V, self.bias_out = self.initialize_LSTM()
                
        # Store loss values
        self.training_loss = []
      
        
        # Define optimizer
        # self.optimizer = torch.optim.Adam([self.W_f, self.W_i, self.W_C, self.W_o, self.U_f, self.U_i, self.U_C, self.U_o, self.b_i, self.b_f, self.b_C, self.b_o, self.V, self.bias_out], lr=1e-3)    
    
    # Initialize network weights and biases using Xavier initialization
    def initialize_LSTM(self):      
        # Xavier initialization
        def xavier_init(size):
            
            in_dim = size[0]
            out_dim = size[1]
            xavier_stddev = np.sqrt(2.0/(in_dim + out_dim))

            return torch.nn.Parameter(xavier_stddev*torch.randn(in_dim, out_dim).type(self.dtype), requires_grad=True)
        
        # vertical weights
        # W_f = Variable(torch.eye(self.hidden_dim).type(self.dtype), requires_grad=True)
        W_f = torch.nn.Parameter(torch.eye(self.hidden_dim).type(self.dtype), requires_grad=True)
        W_i = torch.nn.Parameter(torch.eye(self.hidden_dim).type(self.dtype), requires_grad=True)
        W_C = torch.nn.Parameter(torch.eye(self.hidden_dim).type(self.dtype), requires_grad=True)
        W_o = torch.nn.Parameter(torch.eye(self.hidden_dim).type(self.dtype), requires_grad=True)

        # input weights
        U_f = xavier_init(size=[self.X_dim, self.hidden_dim])
        U_i = xavier_init(size=[self.X_dim, self.hidden_dim])
        U_C = xavier_init(size=[self.X_dim, self.hidden_dim])
        U_o = xavier_init(size=[self.X_dim, self.hidden_dim])

        # biases
        b_i = torch.nn.Parameter(torch.zeros(1,self.hidden_dim).type(self.dtype), requires_grad=True)
        b_f = torch.nn.Parameter(torch.zeros(1,self.hidden_dim).type(self.dtype), requires_grad=True)
        b_C = torch.nn.Parameter(torch.zeros(1,self.hidden_dim).type(self.dtype), requires_grad=True)
        b_o = torch.nn.Parameter(torch.zeros(1,self.hidden_dim).type(self.dtype), requires_grad=True)
        
        V = xavier_init(size=[self.hidden_dim, self.Y_dim])
        bias_out = torch.nn.Parameter(torch.zeros(1,self.Y_dim).type(self.dtype), requires_grad=True)

        torch.nn.init.xavier_normal_(self.fc1.weight)
        torch.nn.init.xavier_normal_(self.fc2.weight)
        
        return W_f, W_i, W_C, W_o, U_f, U_i, U_C, U_o, b_i, b_f, b_C, b_o, V, bias_out
       
           
    # Evaluates the forward pass
    def forward(self, X):
        
        H = torch.zeros(X.shape[1], self.hidden_dim).type(self.dtype).requires_grad_()
        C = torch.zeros(X.shape[1], self.hidden_dim).type(self.dtype).requires_grad_()

        for i in range(0, self.lags):

            # forget gate
            # print(X.dtype)
            # print(H.dtype)
            # print(self.U_f.dtype)
            F = torch.sigmoid( torch.matmul(H, self.W_f) + torch.matmul(X[i,:,:], self.U_f) + self.b_f)
            # input gate
            I = torch.sigmoid( torch.matmul(H, self.W_i) + torch.matmul(X[i,:,:], self.U_i) + self.b_i)

            Cint = torch.tanh( torch.matmul(H, self.W_C) + torch.matmul(X[i,:,:], self.U_C) + self.b_C)
            # cell state
            C = F * C + I * Cint

            # hidden layer
            H = torch.sigmoid( torch.matmul(H,self.W_o) + torch.matmul(X[i,:,:], self.U_o) + self.b_o ) * torch.tanh(C)

        H = torch.matmul(H, self.V ) + self.bias_out

        print(H.shape)
        # out = self.fc1(H)
        out = self.fc2(H)

        return out


class torchLSTM(torch.nn.Module):
    # Initialize the class
    def __init__ (self, lstm_in, lstm_hidden, nlags, lstm_layers = 1):
        super(torchLSTM, self).__init__()

        if torch.cuda.is_available() == True:
            self.dtype_double = torch.cuda.FloatTensor
            self.dtype_int = torch.cuda.LongTensor
        else:
            self.dtype_double = torch.FloatTensor
            self.dtype_int = torch.LongTensor

        self.lstm_in = lstm_in
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers
        self.nlags = nlags

        self.lstm = torch.nn.LSTM(lstm_in, lstm_hidden, num_layers = lstm_layers)

        # sequence length is the number of lags
        # input dimension is the number of time series / features
        # hidden dim is the number of hidden features

        ## can add an outdim, out bias layer here. just a linear layer to map from hidden dim

        self.fc1 = torch.nn.Linear(lstm_hidden, 3)

        torch.nn.init.xavier_normal_(self.fc1.weight)

    def forward(self, X):

        h0 = torch.zeros(self.lstm_layers, X.shape[1], self.lstm_hidden).type(self.dtype_double).requires_grad_() #.cuda()
        c0 = torch.zeros(self.lstm_layers, X.shape[1], self.lstm_hidden).type(self.dtype_double).requires_grad_()

        out, (ht, ct) = self.lstm(X, (h0, c0))

        conv_seq_len = out.size(0)
        batch_size = out.size(1)
        hidden_size = out.size(-1)

        # print(ht.shape)
        # print(out.shape)
        output = torch.tanh(self.fc1(ht.squeeze(0)))
        # output = torch.tanh(self.fc1(out))

        return output