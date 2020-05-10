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
            torch.nn.Conv2d(1, 16, kernel_size=5, padding=2),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
#             torch.nn.MaxPool2d(kernel_size = 2, stride = 2)
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
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
    
        self.fc1 = torch.nn.Linear(32*18*18, 512)
        torch.nn.init.xavier_normal_(self.fc1.weight)
        self.fc2 = torch.nn.Linear(512,64)
        torch.nn.init.xavier_normal_(self.fc2.weight)
        self.fc3 = torch.nn.Linear(64,3)
        torch.nn.init.xavier_normal_(self.fc3.weight)
        
    def forward_pass(self, x):
        out = self.layer1(x)
        # print(out.shape)
        out = self.layer2(out)
        # print(out.shape)
        # out = self.layer3(out)
        out = out.view(out.size(0), -1) # flattens out for all except first dimension ( equiv to np. reshape) for fully connected layer
#         print(out.shape)
        # out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)

        return out

class LSTM:
    # Initialize the class
    def __init__(self, X, Y, hidden_dim):
        
        # Check if there is a GPU available
        if torch.cuda.is_available() == True:
            self.dtype = torch.cuda.FloatTensor
        else:
            self.dtype = torch.FloatTensor
            
        # X has the form lags x data x dim
        # Y has the form data x dim

        # Define PyTorch variables
        X = torch.from_numpy(X).type(self.dtype)
        Y = torch.from_numpy(Y).type(self.dtype)
        self.X = Variable(X, requires_grad=False)
        self.Y = Variable(Y, requires_grad=False)
                 
        self.X_dim = X.shape[-1]
        self.Y_dim = Y.shape[-1]
        self.hidden_dim = hidden_dim
        self.lags = X.shape[0]
        
        # Initialize network weights and biases        
        self.W_f, self.W_i, self.W_C, self.W_o, self.U_f, self.U_i, self.U_C, self.U_o, self.b_i, self.b_f, self.b_C, self.b_o, self.V, self.bias_out = self.initialize_LSTM()
                
        # Store loss values
        self.training_loss = []
      
        # Define optimizer
        self.optimizer = torch.optim.Adam([self.W_f, self.W_i, self.W_C, self.W_o, self.U_f, self.U_i, self.U_C, self.U_o, self.b_i, self.b_f, self.b_C, self.b_o, self.V, self.bias_out], lr=1e-3)    
    
    # Initialize network weights and biases using Xavier initialization
    def initialize_LSTM(self):      
        # Xavier initialization
        def xavier_init(size):
            in_dim = size[0]
            out_dim = size[1]
            xavier_stddev = np.sqrt(2.0/(in_dim + out_dim))
            return Variable(xavier_stddev*torch.randn(in_dim, out_dim).type(self.dtype), requires_grad=True)
        
        # vertical weights
        W_f = Variable(torch.eye(self.hidden_dim).type(self.dtype), requires_grad=True)
        W_i = Variable(torch.eye(self.hidden_dim).type(self.dtype), requires_grad=True)
        W_C = Variable(torch.eye(self.hidden_dim).type(self.dtype), requires_grad=True)
        W_o = Variable(torch.eye(self.hidden_dim).type(self.dtype), requires_grad=True)

        # input weights
        U_f = xavier_init(size=[self.X_dim, self.hidden_dim])
        U_i = xavier_init(size=[self.X_dim, self.hidden_dim])
        U_C = xavier_init(size=[self.X_dim, self.hidden_dim])
        U_o = xavier_init(size=[self.X_dim, self.hidden_dim])

        # biases
        b_i = Variable(torch.zeros(1,self.hidden_dim).type(self.dtype), requires_grad=True)
        b_f = Variable(torch.zeros(1,self.hidden_dim).type(self.dtype), requires_grad=True)
        b_C = Variable(torch.zeros(1,self.hidden_dim).type(self.dtype), requires_grad=True)
        b_o = Variable(torch.zeros(1,self.hidden_dim).type(self.dtype), requires_grad=True)
        
        V = xavier_init(size=[self.hidden_dim, self.Y_dim])
        bias_out = Variable(torch.zeros(1,self.Y_dim).type(self.dtype), requires_grad=True)
        
        return W_f, W_i, W_C, W_o, U_f, U_i, U_C, U_o, b_i, b_f, b_C, b_o, V, bias_out
       
           
    # Evaluates the forward pass
    def forward_pass(self, X):
        
        H = torch.zeros(X.shape[1], self.hidden_dim).type(self.dtype).requires_grad_()
        C = torch.zeros(X.shape[1], self.hidden_dim).type(self.dtype).requires_grad_()

        for i in range(0, self.lags):

            # forget gate
            F = torch.sigmoid( torch.matmul(H, self.W_f) + torch.matmul(X[i,:,:], self.U_f) + self.b_f)
            # input gate
            I = torch.sigmoid( torch.matmul(H, self.W_i) + torch.matmul(X[i,:,:], self.U_i) + self.b_i)

            Cint = torch.tanh( torch.matmul(H, self.W_C) + torch.matmul(X[i,:,:], self.U_C) + self.b_C)
            # cell state
            C = F * C + I * Cint

            # hidden layer
            H = torch.sigmoid( torch.matmul(H,self.W_o) + torch.matmul(X[i,:,:], self.U_o) + self.b_o ) * torch.tanh(C)

        H = torch.matmul(H, self.V ) + self.bias_out
        return H
    
    
    # Computes the mean square error loss
    def compute_loss(self, X, Y):
        loss = torch.mean((Y - self.forward_pass(X))**2)
        return loss
        
    # Fetches a mini-batch of data
    def fetch_minibatch(self,X, y, N_batch):
        N = X.shape[1]
        idx = torch.randperm(N)[0:N_batch]
        X_batch = X[:,idx,:]
        y_batch = y[idx,:]        
        return X_batch, y_batch
    
    
    # Trains the model by minimizing the MSE loss
    def train(self, nIter = 20000, batch_size = 100):
        
        start_time = timeit.default_timer()
        for it in range(nIter):
            # Fetch mini-batch
            X_batch, Y_batch = self.fetch_minibatch(self.X, self.Y, batch_size)
            
            loss = self.compute_loss(X_batch, Y_batch)
            
            # Store loss value
            self.training_loss.append(loss)
            
            # Backward pass
            loss.backward()
            
            # update parameters
            self.optimizer.step()

            # print(self.W_f)
            # print(self.W_o)
            
            # Reset gradients for next step
            self.optimizer.zero_grad()
            
            # Print
            if it % 500 == 0:
                elapsed = timeit.default_timer() - start_time
                print('It: %d, Loss: %.3e, Time: %.2f' % 
                      (it, loss.cpu().data.numpy(), elapsed))
                start_time = timeit.default_timer()
    
    
   # Evaluates predictions at test points    
    def predict(self, X_star):
        X_star = torch.from_numpy(X_star).type(self.dtype)
        y_star = self.forward_pass(X_star)
        y_star = y_star.cpu().data.numpy()
        return y_star