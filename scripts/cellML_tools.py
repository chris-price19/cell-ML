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

# from torch.autograd import Variable,grad
import pandas as pd
from skimage.transform import resize

from zipfile import ZipFile
from PIL import Image

import os
import sys
import re


class single_cell_dataset(Dataset):
    
    def __init__(self, file_location, transformx = None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        if torch.cuda.is_available() == True:
            self.dtype_double = torch.cuda.FloatTensor
            self.dtype_int = torch.cuda.LongTensor
        else:
            self.dtype_double = torch.FloatTensor
            self.dtype_int = torch.LongTensor

        self.zfile = file_location
        with ZipFile(self.zfile, 'r') as zipObj:
        	self.filenames = zipObj.namelist()

        self.filenames = sorted(self.filenames, key=lambda x: ( int(x.split('_')[0].split('c')[-1]), int(x.split('_')[1].split('t')[-1]) ) )
       
        u, c = np.unique([int(ss.split('_')[0].split('c')[-1]) for ss in self.filenames], return_counts = True)
        self.id_map = np.concatenate((u[:,None],c[:,None]), axis=1)

        # print(self.filenames)
        self.zipObj = ZipFile(self.zfile, 'r')

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

class time_coherent_sampler(Sampler):

    def __init__(self, data_source, batch_size = 10):

        self.data_source = data_source
        # self.zfile = file_location
        # with ZipFile(self.zfile, 'r') as zipObj:
        #     self.filenames = zipObj.namelist()

        self.filenames = self.data_source.filenames
        # self.filenames = sorted(self.filenames, key=lambda x: ( int(x.split('_')[0].split('c')[-1]), int(x.split('_')[1].split('t')[-1]) ) )
        # u, ind, c = np.unique([int(ss.split('_')[0].split('c')[-1]) for ss in self.filenames], return_index=True, return_counts = True)
        # self.id_map = np.concatenate((u[:,None], ind[:,None], c[:,None]), axis=1)        
        self.id_map = np.array([int(x.split('_')[0].split('c')[-1]) for x in self.filenames])
        self.unique = np.unique(self.id_map)

        self.batch_size = batch_size

    def __iter__(self):

        n = len(self.unique)
        cinds = torch.randint(high=n, size=(self.batch_size,))
        self.allinds = np.where(np.in1d(self.id_map, cinds))[0].tolist()
        # allinds = tuple(map(tuple, np.where(np.in1d(self.id_map, cinds))[0]))
        # self.batch_size = len(allinds)

        # print(iter(range(len(self.data_source))))
        # print(iter(allinds))
        
        return iter(self.allinds)

    def __len__(self):

        return len(self.allinds)


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
    
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-1)

    def train(self, epochs, bsize = 128):
        
#         bsize = int(np.ceil(len(traindata) * batch_pct))
        # self.train_loader = DataLoader(self.train_data, batch_size=bsize, shuffle=True)
        losslist = []
        
        start_time = timeit.default_timer()
        for epoch in range(epochs):
            
            for it, (images, labels, ids) in enumerate(self.train_data):
                
                self.optimizer.zero_grad()

                # print(type(images))
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
        
        total = len(np.vstack([i[1].detach().numpy() for i in valid_data]))
        print(total)
        u, c = np.unique(total)
        class_dim = len(u)
        # class_dim = len(np.unique([i[1].item() for i in valid_data]))
        print('class balance')
        print(c / np.sum(c))
        c_matrix = np.zeros((class_dim, class_dim))
        
        for it, (images, labels, ids) in enumerate(valid_data):

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
    lossL = cnetmodel.train(epochs = 5, bsize = bsize)