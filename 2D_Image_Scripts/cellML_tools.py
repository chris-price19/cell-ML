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

# from torch.autograd import Variable,grad
import pandas as pd
from skimage.transform import resize

from zipfile import ZipFile
from PIL import Image

from model_components import *

import os
import sys
import re


class single_cell_dataset(Dataset):
    
    def __init__(self, file_location, transformx = None):

        if torch.cuda.is_available() == True:
            self.dtype_double = torch.cuda.FloatTensor
            self.dtype_int = torch.cuda.LongTensor
        else:
            self.dtype_double = torch.FloatTensor
            self.dtype_int = torch.LongTensor

        # if 'chrispr' in os.getcwd():
        #     self.filenames = os.listdir()
        # else:
        with ZipFile(file_location, 'r') as zipObj:
        	self.filenames = zipObj.namelist()

        self.filenames = sorted(self.filenames, key=lambda x: ( int(x.split('_')[0].split('c')[-1]), int(x.split('_')[1].split('t')[-1]) ) )
       
        u, c = np.unique([int(ss.split('_')[0].split('c')[-1]) for ss in self.filenames], return_counts = True)
        self.id_map = np.concatenate((u[:,None],c[:,None]), axis=1)

        # print(self.filenames)
        self.zipObj = ZipFile(file_location, 'r')

        self.transformx = transformx        
        
    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
                
        if torch.is_tensor(idx):
            idx = idx.tolist()

        name = self.filenames[idx]
        img = np.array(Image.open(self.zipObj.open(name)))
        
        dim = 72
        # img = resize(img, (dim, dim), interpolation = cv2.INTER_AREA)
        img = resize(img, (dim, dim), order = 1)
        img = img.astype('float')

        img = np.reshape(img,(1, dim, dim))
 
        ## plastic labels
        nm = name.split("_")
        if nm[2]=="L":
            label = 0
        elif nm[2]=="M":
            label = 1
        elif nm[2]=="H":
            label = 2
        else:
            print("String Parsing Error")
        
        label = np.array(label).astype('float')
        label = np.reshape(label,(1))
        
        if self.transformx:
            img = self.transformx(img).copy()

        cell_ind = int(name.split('_')[0].split('c')[-1])
        time_ind = int(name.split('_')[1].split('t')[-1])

        # return torch.from_numpy(img).type(self.dtype_double), torch.from_numpy(label).type(self.dtype_int), (cell_ind, time_ind)
        # print(img.shape)
        return torch.from_numpy(img), torch.from_numpy(label), (cell_ind, time_ind)

    def get_class_weights(self, pattern):

        weights = np.zeros(len(self.filenames))
        # values = np.zeros(len(pattern))
        for pi, pp in enumerate(pattern):
            # values[pi] = 1. / ''.join(self.filenames).count(pp)
            class_weight = 1. / ''.join(self.filenames).count(pp)

            for fi, ff in enumerate(self.filenames):
                if pp in ff:
                    weights[fi] = class_weight

        return weights

class time_coherent_sampler(Sampler):

    def __init__(self, data_source, bsize = 10):

        self.data_source = data_source
       
        self.filenames = self.data_source.filenames
       
        self.id_map = np.array([int(x.split('_')[0].split('c')[-1]) for x in self.filenames])
        # print(self.id_map)
        self.unique, counts = np.unique(self.id_map, return_counts = True)
       
        self.weights = torch.as_tensor(1. / counts**2)
        self.bsize = bsize

    def __iter__(self):

        n = len(self.unique)
        
        # unweighted
        # cinds = torch.randint(high=n, size=(n,)).numpy()
        ## ############
        # print(cinds)

        # weighted
        ccount = 0
        cinds = torch.multinomial(self.weights, len(self.weights)).numpy()
        self.allinds = []
        for cc in cinds:

            self.allinds.extend(np.where(np.isin(self.id_map, self.unique[cc]))[0].tolist())
            ccount += 1
            
            if ccount == self.bsize:
                # print(self.allinds)
                yield self.allinds
                self.allinds = []
                ccount = 0
        
        if len(self.allinds) > 0:
            yield self.allinds

        # print(np.sum((np.diff(self.allinds)>1)))
        
        # return iter(self.allinds)

    def __len__(self):

        return len(self.allinds)

    
class ConvNet:
    # Initialize the class
    def __init__(self, train_data, valid_data):

        if torch.cuda.is_available() == True:
            self.dtype_double = torch.cuda.FloatTensor
            self.dtype_int = torch.cuda.LongTensor
            self.device = torch.device("cuda:0") 
            # cudnn.benchmark = True
        else:
            self.dtype_double = torch.FloatTensor
            self.dtype_int = torch.LongTensor
            self.device = torch.device("cpu")
        
        self.train_data = train_data
        # pass in data loader
        self.valid_data = valid_data

        self.net = convNN().double().to(self.device) #type(self.dtype_double)

        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
    
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=5e-2)

    def train(self, epochs): #, bsize = 128):
        
#         bsize = int(np.ceil(len(traindata) * batch_pct))
        # self.train_loader = DataLoader(self.train_data, batch_size=bsize, shuffle=True)
        losslist = []
        
        start_time = timeit.default_timer()
        for epoch in range(epochs):

            # scounts = np.zeros(3)
            
            for it, (images, labels, ids) in enumerate(self.train_data):
                
                self.optimizer.zero_grad()

                # scounts += np.unique(labels.numpy(), return_counts = True)[1]

                #   (type(images))
                # print(images.shape)
                # print(images.dtype)
                # print((images.mean(dim=0)).shape)

                # pixel-wise standardization in each batch
                images = (images - images.mean(dim=0)) / (images.std(dim=0) + 1e-8)

                images, labels = images.to(self.device), labels.to(self.device)
                                
                outputs = self.net.forward_pass(images)

                # sys.exit()

                loss = self.loss_fn(outputs, labels.squeeze().long())
                losslist.append(loss)

                loss.backward()
                
                self.optimizer.step()

                if it % 20 == 0:
                    print(losslist[-1])
            
            elapsed = timeit.default_timer() - start_time
            start_time = timeit.default_timer()
            print('Epoch %d: %f s' % (epoch, elapsed))

        return losslist

    def test(self, valid_data):

        correct = 0.
        total = 0.
        loss = 0.
        
        total = np.vstack([i[1].detach().numpy() for i in valid_data])
        u, c = np.unique(total, return_counts=True)
        print(u)
        print(c)
        total = len(total)
        class_dim = len(u)
        print(class_dim)
        # class_dim = len(np.unique([i[1].item() for i in valid_data]))
        print('class balance')
        print(c / np.sum(c))

        c_matrix = np.zeros((class_dim, class_dim))
        
        for it, (images, labels, ids) in enumerate(valid_data):

            images = ((images - images.mean(dim=0)) / (images.std(dim=0) + 1e-8))

            images, labels = images.to(self.device), labels.to(self.device)

            outputs = self.net.forward_pass(images)

            loss = self.loss_fn(outputs, labels.squeeze().long())
#             print(loss)
            _, predicted = torch.max(outputs, dim=1) # .detach().numpy()

            labels = labels.cpu().detach().numpy()
            predicted = predicted.unsqueeze(1).cpu().detach().numpy()
            correct += (predicted == labels).sum()
            # print(type(correct))
            
            for ai in np.arange(class_dim):
                for pi in np.arange(class_dim):
                    c_matrix[pi,ai] += np.sum([(predicted == ai) & (labels == pi)])

        # print(correct)
        # print(total)
        
        return loss, total, (correct / total), c_matrix



class ConvPlusLSTM:
    # Initialize the class
    def __init__(self, train_data, valid_data, nlags, hidden_dim):

        if torch.cuda.is_available() == True:
            self.dtype_double = torch.cuda.FloatTensor
            self.dtype_int = torch.cuda.LongTensor
            self.device = torch.device("cuda:0") 
            # cudnn.benchmark = True
        else:
            self.dtype_double = torch.FloatTensor
            self.dtype_int = torch.LongTensor
            self.device = torch.device("cpu")
        
        # print(self.dtype_double)
        self.train_data = train_data
        # pass in data loader
        self.valid_data = valid_data

        self.cnet = convOnly().float().to(self.device) #type(self.dtype_double)
        
        self.x_dim = 64
        self.y_dim = 32
        self.nlags = nlags

        self.lstm = LSTM(self.x_dim, self.y_dim, self.nlags, hidden_dim).to(self.device)

        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
    
        self.combo_optimizer = (
            torch.optim.Adam(list(self.cnet.parameters()) + list(self.lstm.parameters())
            + [self.lstm.W_f, self.lstm.W_i, self.lstm.W_C, self.lstm.W_o, self.lstm.U_f, self.lstm.U_i, self.lstm.U_C, self.lstm.U_o, self.lstm.b_i, self.lstm.b_f, self.lstm.b_C, self.lstm.b_o, self.lstm.V, self.lstm.bias_out], lr=1e-2)
        )

    def train(self, epochs): #, bsize = 128):
        
#         bsize = int(np.ceil(len(traindata) * batch_pct))
        # self.train_loader = DataLoader(self.train_data, batch_size=bsize, shuffle=True)
        losslist = []
        
        start_time = timeit.default_timer()
        for epoch in range(epochs):
     
            for it, (images, labels, ids) in enumerate(self.train_data):
                
                unq, cnt = np.unique(ids[0], return_counts=True)
                batch_lags = np.sum(cnt - self.nlags+1)
                # print(batch_lags)

                self.combo_optimizer.zero_grad()

                # pixel-wise standardization in each batch
                images = ((images - images.mean(dim=0)) / (images.std(dim=0) + 1e-8)).float()

                images, labels = images.to(self.device), labels.to(self.device)

                # print(images.dtype)
                                
                outputs = self.cnet.forward_pass(images)

                restack = []
                laglabels = []
                
                li = 0
                for ui, uu in enumerate(unq):
                    ci = 0
                    while ci < cnt[ui] - self.nlags + 1:
                        restack.append(outputs[li:li+self.nlags,:])
                        laglabels.append(labels[li])
                        ci += 1
                        li += 1
                    li += self.nlags-1
                # print(restack)
                laglabels = torch.stack(laglabels)
                inputs = torch.stack(restack, dim = 1)
                # print(inputs.shape)
                LSTMoutputs = self.lstm.forward_pass(inputs)
                # print(LSTMoutputs.shape)

                loss = self.loss_fn(LSTMoutputs, laglabels.squeeze().long())
                
                loss.backward()
                losslist.append(loss.detach_())

                del loss
                del LSTMoutputs
                del inputs
                del outputs
                del laglabels

                self.combo_optimizer.step()

                if it % 20 == 0:
                    print(losslist[-1])
            
            elapsed = timeit.default_timer() - start_time
            start_time = timeit.default_timer()
            print('Epoch %d: %f s' % (epoch, elapsed))

        return losslist

    def test(self, valid_data):

        correct = 0.
        total = 0.
        loss = 0.
        
        total = np.vstack([i[1].detach().numpy() for i in valid_data])
        u, c = np.unique(total, return_counts=True)
        print(u)
        print(c)
        total = len(total)
        class_dim = len(u)
        print(class_dim)
        # class_dim = len(np.unique([i[1].item() for i in valid_data]))
        print('class balance')
        print(c / np.sum(c))

        c_matrix = np.zeros((class_dim, class_dim))
        
        for it, (images, labels, ids) in enumerate(valid_data):
            
            unq, cnt = np.unique(ids[0], return_counts=True)
            batch_lags = np.sum(cnt - self.nlags+1)
            # pixel-wise standardization in each batch
            images = ((images - images.mean(dim=0)) / (images.std(dim=0) + 1e-8)).float()

            images, labels = images.to(self.device), labels.to(self.device)
                            
            outputs = self.cnet.forward_pass(images)

            restack = []
            laglabels = []
            
            li = 0
            for ui, uu in enumerate(unq):
                ci = 0
                while ci < cnt[ui] - self.nlags + 1:
                    restack.append(outputs[li:li+self.nlags,:])
                    laglabels.append(labels[li])
                    ci += 1
                    li += 1
                li += self.nlags-1
            # print(restack)
            laglabels = torch.stack(laglabels)
            inputs = torch.stack(restack, dim = 1)

            LSTMoutputs = self.lstm.forward_pass(inputs)

            loss = self.loss_fn(LSTMoutputs, laglabels.squeeze().long())
#             print(loss)
            _, predicted = torch.max(LSTMoutputs, dim=1) # .detach().numpy()

            labels = laglabels.cpu().detach().numpy()
            predicted = predicted.unsqueeze(1).cpu().detach().numpy()
            correct += (predicted == labels).sum()
            # print(type(correct))
            
            for ai in np.arange(class_dim):
                for pi in np.arange(class_dim):
                    c_matrix[pi,ai] += np.sum([(predicted == ai) & (labels == pi)])

        # print(correct)
        # print(total)
        
        return loss, total, (correct / total), c_matrix





if __name__ == "__main__":
	
    pd.set_option('display.expand_frame_repr', False, 'display.max_columns', None)
    cwd = os.getcwd()

    if 'Chris Price' in cwd:
        datadir = 'C:\\Users\\Chris Price\\Box Sync\\Plasticity_Protrusions_ML\\'
    elif 'ccarr' in cwd:
        datadir = 'C:\\Users\\ccarr_nj7afa9\\Box Sync\\Plasticity_Protrusions_ML\\'
    else:
        print('add your path to Plasticity_Protrusions_ML here')
        sys.exit()

    bsize = 256
# ################################# random sampler
#     cell_set = single_cell_dataset(datadir +'images\\data-images\\train_series.zip')
#     test_randloader = DataLoader(cell_set, batch_size=bsize, shuffle=True)

    train_set = single_cell_dataset(datadir +'images\\data-images\\train_series.zip')
    test_set = single_cell_dataset(datadir +'images\\data-images\\test_series.zip')

    train_randloader = DataLoader(train_set, batch_size=bsize, shuffle=True)
    test_randloader = DataLoader(test_set, batch_size=bsize, shuffle=True)

    # for it, (images, labels, ids) in enumerate(test_randloader):
    #     if it < 1:
    #         print(ids)
    #     else:
    #         break
####################################   

# ############################## time sampler
#     num_lags = 6

#     cohsample = time_coherent_sampler(cell_set, batch_size = int(bsize/num_lags))
#     # this is pretty garbage, but as long as batch_size here is large enough relative to batch_size in the sampler, you will get the complete time series returned for all the cells.
#     # needs a better implementation with sampler, batch_sampler. or honestly just a batching class that's not a dataloader.
#     test_cohloader = DataLoader(cell_set, sampler=cohsample, batch_size = 1000)

#     for it, (images, labels, ids) in enumerate(test_cohloader):

#         if it < 1:
#             print(len(np.unique(ids[0])))
#             print(len(ids[0]))
#             print(np.unique(ids[0]))
#             print(ids[1])
#         else:
#             break
# ###################################
	
########### test basic CNN

    cnetmodel = ConvNet(train_randloader, test_randloader) #, testdata)
    lossL = cnetmodel.train(epochs = 5)
