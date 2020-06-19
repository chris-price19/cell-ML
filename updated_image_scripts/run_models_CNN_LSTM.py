#!/usr/bin/python

"""
Version 4/14/2020
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

from lib import cellML_tools as cmt
# from libPARA import cellML_tools as cmt

def train_test_split(datadir, fs):

    #### make train and test datasets
    data_fraction = 0.05 # to speed up testing, use some random fraction of images.
    train_fraction = 0.85
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

	# from libPARA import cellML_tools as cmt

    gpu = args.local_rank
               
    dist.init_process_group(                                   
        backend='nccl',                                         
        init_method='env://'                                           
    )
    
    torch.cuda.set_device(gpu)
    torch.cuda.manual_seed(9192)

    bsize = 20
    img_dim = 72

    nlags = 6
    hidden = 128

    train_set = cmt.single_cell_dataset(datadir + 'train' + fs, img_dim)
    test_set = cmt.single_cell_dataset(datadir + 'test' + fs, img_dim)

    if torch.cuda.is_available() == True:
        pinning = True
    else:
        pinning = False


    train_sampler = cmt.distributed_balanced_time_coherent_sampler(train_set, nlags, bsize, num_replicas=args.world_size, rank=args.local_rank)
    
    train_loader = DataLoader(train_set, batch_sampler = train_sampler, pin_memory = pinning)

    # laglabels = []
    # batch_lags = 0
    # for it, (images, labels, ids) in enumerate(train_loader):
    	
    #     unq, cnt = np.unique(ids[0], return_counts=True)
    #     batch_lags += np.sum(cnt - nlags+1)
    #     li = 0

    #     for ui, uu in enumerate(unq):
    #         ci = 0
    #         while ci < cnt[ui] - nlags + 1:
    #             # restack.append(outputs[li:li+nlags,:])
    #             laglabels.extend(labels[li])
    #             ci += 1
    #             li += 1
    #         li += nlags-1
    #     # print(restack)
    #     # laglabels2 = torch.stack(laglabels)
    # print(batch_lags)
    # u, c = np.unique(laglabels, return_counts=True)
    # # print(u)
    # # print(c)
    # print(c / np.sum(c))

    test_sampler = cmt.distributed_balanced_time_coherent_sampler(test_set, nlags, bsize, num_replicas=args.world_size, rank=args.local_rank)
    test_loader = DataLoader(test_set, batch_sampler = test_sampler, pin_memory = pinning)

    # sys.exit()

    # ########### test combined CNN LSTM

    cnetLSTMmodel = cmt.ConvPlusLSTM(train_loader, test_loader, nlags, hidden)
    cnetLSTMmodel.cnet.cuda(gpu)
    cnetLSTMmodel.lstm.cuda(gpu)
    cnetLSTMmodel.cnet = torch.nn.parallel.DistributedDataParallel(cnetLSTMmodel.cnet, device_ids=[gpu], find_unused_parameters=True)
    cnetLSTMmodel.lstm = torch.nn.parallel.DistributedDataParallel(cnetLSTMmodel.lstm, device_ids=[gpu], find_unused_parameters=True)

    lossL = cnetLSTMmodel.train(250, len(train_set))

    s1, s2, s3 = cnetLSTMmodel.test(cnetLSTMmodel.train_data)

    print('end training data acc check')

    s1, s2, s3 = cnetLSTMmodel.test(cnetLSTMmodel.valid_data)

    print('end testing data acc check')

    # print('training acc = %f' % train_frac)
    # print('testing acc = %f' % test_frac)

    # print('confusion matrix')
    # print(c_matrix)
    if dist.get_rank() == 0:
        torch.save(cnetLSTMmodel.cnet.state_dict(), './convOnly_trained.out')
        torch.save(cnetLSTMmodel.lstm.state_dict(), './LSTM_trained.out')

    return


def serial_run(datadir, fs):

	# print(torch.cuda.device_count())
    # print(torch.cuda.current_device())
    # print(torch.cuda.get_device_name(0))
    # torch.cuda.device()

    # sys.exit()
    

    bsize = 5
    img_dim = 72

    nlags = 6
    hidden = 128

    train_set = cmt.single_cell_dataset(datadir + 'train' + fs, img_dim)
    test_set = cmt.single_cell_dataset(datadir + 'test/', img_dim)

    if torch.cuda.is_available() == True:
        pinning = True
    else:
        pinning = False


    train_coherentsampler = cmt.time_coherent_sampler(train_set, nlags, bsize = bsize)
    # print(train_coherentsampler.counts)
    test_coherentsampler = cmt.time_coherent_sampler(test_set, nlags, bsize = bsize)

    train_loader = DataLoader(train_set, batch_sampler = train_coherentsampler, pin_memory = pinning)
    test_loader = DataLoader(test_set, batch_sampler = test_coherentsampler, pin_memory = pinning)

    # total = np.vstack([i[1].detach().numpy() for i in train_loader])
    laglabels = []
    batch_lags = 0
    for it, (images, labels, ids) in enumerate(train_loader):
    	
        unq, cnt = np.unique(ids[0], return_counts=True)
        batch_lags += np.sum(cnt - nlags+1)
        li = 0

        for ui, uu in enumerate(unq):
            ci = 0
            while ci < cnt[ui] - nlags + 1:
                # restack.append(outputs[li:li+nlags,:])
                laglabels.extend(labels[li])
                ci += 1
                li += 1
            li += nlags-1
        # print(restack)
        # laglabels2 = torch.stack(laglabels)
    print(batch_lags)
    u, c = np.unique(laglabels, return_counts=True)
    # print(u)
    # print(c)
    print(c / np.sum(c))
    

    cnetLSTMmodel = cmt.ConvPlusLSTM(train_loader, test_loader, nlags, hidden)

    lossL = cnetLSTMmodel.train(1, len(train_set))

    trainloss, vsize, train_frac, c_matrix = cnetLSTMmodel.test(cnetLSTMmodel.train_data)
    testloss, vsize, test_frac, c_matrix = cnetLSTMmodel.test(cnetLSTMmodel.valid_data)

    print('training acc = %f' % train_frac)
    print('testing acc = %f' % test_frac)

    print('confusion matrix')
    print(c_matrix)


    return


def main(datadir, fs):

	# parallel = True
	parallel = False

	if parallel:

		if torch.cuda.is_available() == True:

			# from libPARA import cellML_tools as cmt

			print('device count')
			print(torch.cuda.device_count())
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
	else:

		# from lib import cellML_tools as cmt

		print('no gpu access, quitting')
		# sys.exit()
		serial_run(datadir, fs)

	return

if __name__ == '__main__':

    pd.set_option('display.expand_frame_repr', False, 'display.max_columns', None)
    cwd = os.getcwd()

    if 'Chris Price' in cwd:
        datadir = 'C:\\Users\\Chris Price\\Box Sync\\Plasticity_Protrusions_ML\\images\\data-images\\'
        fs = '\\'
    elif 'ccarr' in cwd:
        datadir = 'C:\\Users\\ccarr_nj7afa9\\Box Sync\\Plasticity_Protrusions_ML\\images\\data-images\\'
        fs = '\\'
    elif 'chrispr' in cwd:    
        datadir = '/home/chrispr/mem/chrispr/ml_cell/images/data-images/'
        fs = '/'
        # sys.stdout = open('log.txt', 'w')
    else:
        print('add your path to Plasticity_Protrusions_ML here')
        sys.exit()

    train_test_split(datadir, fs)

    main(datadir, fs)