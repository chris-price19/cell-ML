#!/usr/bin/python

"""
Version 4/14/2020

modules for CNN, RNN, LSTM, data loader for segmented cells out of zip file.

"""


import numpy as np
import scipy

import matplotlib.pyplot as plt

import torch
from torch.utils.data import Sampler, Dataset, DataLoader, ConcatDataset, WeightedRandomSampler, RandomSampler
from torchvision import transforms, utils
import timeit

from torch.autograd import Variable
import pandas as pd
from skimage.transform import resize

from zipfile import ZipFile
from PIL import Image

import os
import sys
import re

from cellML_tools import *

pd.set_option('display.expand_frame_repr', False, 'display.max_columns', None)
cwd = os.getcwd()

if 'Chris Price' in cwd:
    datadir = 'C:\\Users\\Chris Price\\Box Sync\\Plasticity_Protrusions_ML\\'
    fs = '\\'
elif 'ccarr' in cwd:
    datadir = 'C:\\Users\\ccarr_nj7afa9\\Box Sync\\Plasticity_Protrusions_ML\\'
    fs = '\\'
elif 'chrispr' in cwd:    
    datadir = '/home/chrispr/mem/chrispr/ml_cell/'
    fs = '/'
    # sys.stdout = open('log.txt', 'w')
else:
    # print('add your path to Plasticity_Protrusions_ML here')
    # sys.exit()
    print('using sample_data folder if it exists')

################################# random sampler, pure CNN

#### make train and test datasets
data_fraction = 0.4 # to speed up testing, use some random fraction of images.
train_fraction = 0.85
test_fraction = 1 - train_fraction
# np.random.seed(11820)

# with ZipFile(datadir +'images' + fs + 'data-images' + fs + 'cell_series.zip', 'r') as zipObj:
#     filenames = zipObj.namelist()

with ZipFile('./sample_data/cell_series.zip', 'r') as zipObj:
    filenames = zipObj.namelist()

# test = [filenames[2], filenames[37000], filenames[40000]]
# pattern = ['_L', '_M','_H']
# print([p for p in pattern if p in [tt for tt in test]])

# print(test)
# print(''.join(filenames).count('_L'))
# print(''.join(filenames).count('_M'))
# print(''.join(filenames).count('_H'))

filenames = sorted(filenames, key=lambda x: ( int(x.split('_')[0].split('c')[-1]), int(x.split('_')[1].split('t')[-1]) ) )
u, c = np.unique([int(ss.split('_')[0].split('c')[-1]) for ss in filenames], return_counts = True)
id_map = np.concatenate((u[:,None],c[:,None]), axis=1)

np.random.shuffle(id_map)

ushuff = id_map[:int(len(id_map)*data_fraction),0]

utrain = ushuff[:int(len(ushuff) * train_fraction)].tolist()
utest = ushuff[int(len(ushuff) * train_fraction):].tolist()

trainfiles = [f for f in filenames if int(f.split('_')[0].split('c')[-1]) in utrain]
testfiles = [f for f in filenames if int(f.split('_')[0].split('c')[-1]) in utest]

zipIn = ZipFile(open('./sample_data/cell_series.zip', 'rb'))
zipTrain = ZipFile('./sample_data/train_series.zip', 'w')
zipTest = ZipFile('./sample_data/test_series.zip', 'w')

for ti, tt in enumerate(trainfiles):

	zipTrain.writestr(tt, zipIn.read(tt))

zipTrain.close()

for ti, tt in enumerate(testfiles):

	zipTest.writestr(tt, zipIn.read(tt))

zipTest.close()
zipIn.close()
#################################


########### test basic CNN

bsize = 256 * 2
pattern = ['_L','_M','_H']

# train_set = single_cell_dataset(datadir +'images' + fs + 'data-images' + fs + 'train_series.zip')
train_set = single_cell_dataset('./sample_data/train_series.zip')
wghts = train_set.get_class_weights(pattern)
# print(wghts[0:10])
# print(wghts[20000])
# sys.exit()
train_sampler = WeightedRandomSampler(torch.from_numpy(wghts), len(wghts))

# test_set = single_cell_dataset(datadir +'images' + fs + 'data-images' + fs + 'test_series.zip')
test_set = single_cell_dataset('./sample_data/test_series.zip')
wghts = test_set.get_class_weights(pattern)

test_sampler = WeightedRandomSampler(torch.from_numpy(wghts), len(wghts))

if torch.cuda.is_available() == True:
    pinning = True
else:
    pinning = False

train_randloader = DataLoader(train_set, batch_size=bsize, sampler = train_sampler, pin_memory = pinning) #, num_workers= 4) #  shuffle=True,
test_randloader = DataLoader(test_set, batch_size=bsize,  sampler = test_sampler, pin_memory = pinning) #, num_workers=4) # shuffle=True,

cnetmodel = ConvNet(train_randloader, test_randloader)
lossL = cnetmodel.train(epochs = 300)

trainloss, vsize, train_frac, c_matrix = cnetmodel.test(cnetmodel.train_data)
testloss, vsize, test_frac, c_matrix = cnetmodel.test(cnetmodel.valid_data)

print('training acc = %f' % train_frac)
print('testing acc = %f' % test_frac)

print('confusion matrix')
print(c_matrix)



# ### adding time series below
# #################################

# bsize = 20

# train_set = single_cell_dataset(datadir +'images' + fs + 'data-images' + fs + 'train_series.zip')
# test_set = single_cell_dataset(datadir +'images' + fs + 'data-images' + fs + 'test_series.zip')

# print(len(train_set))

# if torch.cuda.is_available() == True:
#     pinning = True

#     # self.dtype_double = torch.cuda.FloatTensor
#     # self.dtype_int = torch.cuda.LongTensor
#     # self.device = torch.device("cuda:0")            
# else:
#     pinning = False
#     # self.dtype_double = torch.FloatTensor
#     # self.dtype_int = torch.LongTensor
#     # self.device = torch.device("cpu")


# train_coherentsampler = time_coherent_sampler(train_set, bsize = bsize)
# test_coherentsampler = time_coherent_sampler(test_set,  bsize = bsize)

# train_loader = DataLoader(train_set, batch_sampler = train_coherentsampler, pin_memory = pinning)
# test_loader = DataLoader(test_set, batch_sampler = test_coherentsampler, pin_memory = pinning)

# ###################################

# # for ii, (images, labels, ids) in enumerate(train_loader):
    
# #     print(ii)


# # ########### test combined CNN LSTM
# # ### need new model class here.
# nlags = 6
# hidden = 128

# cnetLSTMmodel = ConvPlusLSTM(train_loader, test_loader, nlags, hidden)

# lossL = cnetLSTMmodel.train(epochs = 250)

# trainloss, vsize, train_frac, c_matrix = cnetLSTMmodel.test(cnetLSTMmodel.train_data)
# testloss, vsize, test_frac, c_matrix = cnetLSTMmodel.test(cnetLSTMmodel.valid_data)

# print('training acc = %f' % train_frac)
# print('testing acc = %f' % test_frac)

# print('confusion matrix')
# print(c_matrix)
