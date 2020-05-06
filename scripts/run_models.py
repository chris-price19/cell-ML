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
else:
    print('add your path to Plasticity_Protrusions_ML here')
    sys.exit()

################################# random sampler, pure CNN

#### make train and test datasets
data_fraction = 0.5 # to speed up testing, use some random fraction of images.
train_fraction = 0.85
test_fraction = 1 - train_fraction
np.random.seed(11820)

with ZipFile(datadir +'images' + fs + 'data-images' + fs + 'cell_series.zip', 'r') as zipObj:
    filenames = zipObj.namelist()

filenames = sorted(filenames, key=lambda x: ( int(x.split('_')[0].split('c')[-1]), int(x.split('_')[1].split('t')[-1]) ) )
u, c = np.unique([int(ss.split('_')[0].split('c')[-1]) for ss in filenames], return_counts = True)
id_map = np.concatenate((u[:,None],c[:,None]), axis=1)

np.random.shuffle(id_map)

ushuff = id_map[:int(len(id_map)*data_fraction),0]

utrain = ushuff[:int(len(ushuff) * train_fraction)].tolist()
utest = ushuff[int(len(ushuff) * train_fraction):].tolist()

trainfiles = [f for f in filenames if int(f.split('_')[0].split('c')[-1]) in utrain]
testfiles = [f for f in filenames if int(f.split('_')[0].split('c')[-1]) in utest]

zipIn = ZipFile(open(datadir + 'images' + fs + 'data-images' + fs + 'cell_series.zip', 'rb'))
zipTrain = ZipFile(datadir +'images' + fs + 'data-images' + fs + 'train_series.zip', 'w')
zipTest = ZipFile(datadir +'images' + fs + 'data-images' + fs + 'test_series.zip', 'w')

for ti, tt in enumerate(trainfiles):

	zipTrain.writestr(tt, zipIn.read(tt))

zipTrain.close()

for ti, tt in enumerate(testfiles):

	zipTest.writestr(tt, zipIn.read(tt))

zipTest.close()
#################################

bsize = 256 * 2

train_set = single_cell_dataset(datadir +'images' + fs + 'data-images' + fs + 'train_series.zip')

test_set = single_cell_dataset(datadir +'images' + fs + 'data-images' + fs + 'test_series.zip')
if torch.cuda.is_available() == True:
    pinning = True
else:
    pinning = False

train_randloader = DataLoader(train_set, batch_size=bsize, shuffle=True, pin_memory = pinning, num_workers=4)
test_randloader = DataLoader(test_set, batch_size=bsize, shuffle=True, pin_memory = pinning, num_workers=4)

# print([i[1].detach().numpy() for i in test_randloader])

# sys.exit()


########### test basic CNN

cnetmodel = ConvNet(train_randloader, test_randloader)
lossL = cnetmodel.train(epochs = 50, bsize = bsize)

trainloss, vsize, train_frac, c_matrix = cnetmodel.test(cnetmodel.train_data)
testloss, vsize, test_frac, c_matrix = cnetmodel.test(cnetmodel.valid_data)

print('training acc = %f' % train_frac)
print('testing acc = %f' % test_frac)

print('confusion matrix')
print(c_matrix)



# ### adding time series below
# #################################

# bsize = 10

# train_set = single_cell_dataset(datadir +'images' + fs + 'data-images' + fs + 'train_series.zip')

# test_set = single_cell_dataset(datadir +'images' + fs + 'data-images' + fs + 'test_series.zip')
# if torch.cuda.is_available() == True:
#     pinning = True
# else:
#     pinning = False

# train_coherentsampler = time_coherent_sampler(datadir +'images' + fs + 'data-images' + fs + 'train_series.zip')
# test_coherentsampler = time_coherent_sampler(datadir +'images' + fs + 'data-images' + fs + 'test_series.zip')

# train_loader = DataLoader(train_set, batch_size=bsize*100, sampler=train_coherentsampler, pin_memory = pinning)
# test_loader = DataLoader(test_set, batch_size=bsize*100, sampler=test_coherentsampler, pin_memory = pinning)

# # print([i[1].detach().numpy() for i in test_randloader])

# ########### test combined CNN LSTM
# ### need new model class here.

# cnetmodel = ConvNet(train_randloader, test_randloader)
# lossL = cnetmodel.train(epochs = 10, bsize = bsize)

# trainloss, vsize, train_frac, c_matrix = cnetmodel.test(cnetmodel.train_data)
# testloss, vsize, test_frac, c_matrix = cnetmodel.test(cnetmodel.valid_data)

# print('training acc = %f' % train_frac)
# print('testing acc = %f' % test_frac)

# print('confusion matrix')
# print(c_matrix)