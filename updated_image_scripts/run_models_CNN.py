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
import argparse
import torch.multiprocessing as mp
import torchvision
import torch.distributed as dist

import timeit

from torch.autograd import Variable
import pandas as pd
from skimage.transform import resize

from zipfile import ZipFile
from PIL import Image

import os
import sys
import re

from libPARA import cellML_tools as cmt
# from libPARA import model_components as mc

def train_test_split(datadir, fs):

    #### make train and test datasets
    data_fraction = 1 # to speed up testing, use some random fraction of images.
    train_fraction = 0.8
    test_fraction = 1 - train_fraction
    # np.random.seed(11820)

    with ZipFile(datadir + 'cell_series.zip', 'r') as zipObj:
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

    os.system('rm -rf '+ datadir + 'train')
    os.system('mkdir ' + datadir + 'train')

    os.system('rm -rf '+ datadir + 'test')
    os.system('mkdir ' + datadir + 'test')

    zipIn = ZipFile(open(datadir + 'cell_series.zip', 'rb'))

    for ti, tt in enumerate(trainfiles):

        zipIn.extract(tt, path=datadir + 'train')

    for ti, tt in enumerate(testfiles):

        zipIn.extract(tt, path=datadir + 'test')

    zipIn.close()

    return


def parallel_run(args, datadir):

    gpu = args.local_rank
               
    dist.init_process_group(                                   
        backend='nccl',                                         
        init_method='env://'                                           
    )
    
    torch.cuda.set_device(gpu)
    torch.cuda.manual_seed(9192)

    bsize = 256
    pattern = ['_L','_M','_H']
    img_dim = 96

    train_set = cmt.single_cell_dataset(datadir + 'train/', img_dim)
    wghts = train_set.get_class_weights(pattern)

    # train_sampler = WeightedRandomSampler(torch.from_numpy(wghts), len(wghts))
    # train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, num_replicas=args.world_size, rank=args.local_rank)
    train_sampler = cmt.WeightedDistributedSampler(train_set, wghts, num_replicas=args.world_size, rank=args.local_rank)

    test_set = cmt.single_cell_dataset(datadir + 'test/', img_dim)
    wghts = test_set.get_class_weights(pattern)
    
    test_sampler = cmt.WeightedDistributedSampler(test_set, wghts, num_replicas=args.world_size, rank=args.local_rank)

    train_randloader = DataLoader(train_set, batch_size=bsize, sampler = train_sampler, pin_memory = True)#, num_workers= 4) #  shuffle=True,

    # total = np.vstack([i[1].detach().numpy() for i in valid_data])
    # u, c = np.unique(total, return_counts=True)
    # print('class balance training')
    # print(c / np.sum(c))

    test_randloader = DataLoader(test_set, batch_size=bsize,  sampler = test_sampler, pin_memory = True)#, num_workers=4) # shuffle=True,

    cnetmodel = cmt.ConvNet(train_randloader, test_randloader, 5e-2)
    cnetmodel.net.cuda(gpu)
    cnetmodel.net = torch.nn.parallel.DistributedDataParallel(cnetmodel.net, device_ids=[gpu])
    
    lossL = cnetmodel.train(250, len(train_set))

    # print(len(lossL))
    # print(len(accL))

    s1, s2, s3 = cnetmodel.test(cnetmodel.train_data)

    print('end training data acc check')
    print('\n')
    print('##############')

    s1, s2, s3 = cnetmodel.test(cnetmodel.valid_data)

    print('end testing data acc check')

    # print('training acc = %f' % train_frac)
    # print('testing acc = %f' % test_frac)

    # print('confusion matrix')
    # print(c_matrix)
    if dist.get_rank() == 0:
        torch.save(cnetmodel.net.state_dict(), './convNet_trained.out')

    return

def main(datadir):

    if torch.cuda.is_available() == True:
        print('device count')
        print(torch.cuda.device_count())
    else:
        print('no gpu access, quitting')
        sys.exit()

    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N')
    parser.add_argument('-g', '--gpus', default=2, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument("--local_rank", type=int)
    parser.add_argument('--epochs', default=1, type=int, metavar='N',
                        help='number of total epochs to run')

    args = parser.parse_args()

    args.world_size = args.gpus * args.nodes                #
    
    print(args)
   
    torch.cuda.set_device(args.local_rank)

    parallel_run(args, datadir)

    return


################################# random sampler, pure CNN

########### test basic CNN

if __name__ == '__main__':

    pd.set_option('display.expand_frame_repr', False, 'display.max_columns', None)
    cwd = os.getcwd()

    if 'Chris Price' in cwd:
        datadir = 'C:\\Users\\Chris Price\\Box Sync\\Plasticity_Protrusions_ML\\'
        fs = '\\'
    elif 'ccarr' in cwd:
        datadir = 'C:\\Users\\ccarr_nj7afa9\\Box Sync\\Plasticity_Protrusions_ML\\'
        fs = '\\'
    elif 'chrispr' in cwd:    
        datadir = '/home/chrispr/mem/chrispr/ml_cell/images/data-images/'
        fs = '/'
        # sys.stdout = open('log.txt', 'w')
    else:
        # print('add your path to Plasticity_Protrusions_ML here')
        # sys.exit()
        print('using sample_data folder if it exists')

    train_test_split(datadir, fs)

    main(datadir)





