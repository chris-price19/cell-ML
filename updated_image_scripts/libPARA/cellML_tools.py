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
import torch.distributed as dist
import timeit

# from torch.autograd import Variable,grad
import pandas as pd
from skimage.transform import resize

from zipfile import ZipFile
from PIL import Image

from libPARA import model_components as mc

import os
import sys
import re


class single_cell_dataset(Dataset):
    
    def __init__(self, file_location, dim, transformx = None):

        if torch.cuda.is_available() == True:
            self.dtype_double = torch.cuda.FloatTensor
            self.dtype_int = torch.cuda.LongTensor
        else:
            self.dtype_double = torch.FloatTensor
            self.dtype_int = torch.LongTensor

        self.dim = dim

        self.dir = file_location

        self.filenames = os.listdir(file_location)
        self.filenames = sorted(self.filenames, key=lambda x: ( int(x.split('_')[0].split('c')[-1]), int(x.split('_')[1].split('t')[-1]) ) )
       
        u, c = np.unique([int(ss.split('_')[0].split('c')[-1]) for ss in self.filenames], return_counts = True)
        self.id_map = np.concatenate((u[:,None],c[:,None]), axis=1)

        self.transformx = transformx        
        
    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
                
        if torch.is_tensor(idx):
            idx = idx.tolist()

        name = self.filenames[idx]
        img = np.array(Image.open(self.dir + name))
        
        # img = resize(img, (dim, dim), interpolation = cv2.INTER_AREA)
        img = resize(img, (self.dim, self.dim), order = 1)
        img = img.astype('float')
        img = np.reshape(img,(1, self.dim, self.dim))
 
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

class WeightedDistributedSampler(Sampler):

    def __init__(self, dataset, weights, num_replicas=None, rank=None, shuffle=True):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()

        self.dataset = dataset
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(np.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        if self.shuffle:
            # indices = torch.randperm(len(self.dataset), generator=g).tolist()
            indices = torch.multinomial(self.weights, len(self.weights), replacement=True, generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))


        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


class time_coherent_sampler(Sampler):

    def __init__(self, data_source, nlags, bsize = 10):

        self.data_source = data_source
       
        self.filenames = self.data_source.filenames
       
        self.id_map = np.array([[int(x.split('_')[0].split('c')[-1]), x.split('_')[2]] for x in self.filenames])
        # print(self.id_map)
        pattern = np.unique(self.id_map[:,1])
        # print(pattern)
        self.unique, counts = np.unique(self.id_map, axis=0, return_counts = True)
        # print(self.unique)
        # print(counts)
        # unique is unique cell IDs, counts is number of instances of each cell ID
        self.weights = np.zeros(len(self.unique))
        # tlags = 0
        for pi, pp in enumerate(pattern):
            # values[pi] = 1. / ''.join(self.filenames).count(pp)
            class_weight = 1. / np.sum(counts[self.unique[:,1] == pp] - nlags + 1)
            # tlags += np.sum(counts[self.unique[:,1] == pp] - nlags + 1)
            self.weights[self.unique[:,1] == pp] = class_weight
        
        # print(tlags)
        # print(self.weights)
        self.unique = self.unique[:,0].astype(np.int32)
        self.id_map = self.id_map[:,0].astype(np.int32)
        # print(self.unique)
        self.weights = torch.as_tensor(self.weights)
        self.bsize = bsize

    def __iter__(self):

        # g = torch.Generator()
        # g.manual_seed(0)
        # weighted
        ccount = 0
        cinds = torch.multinomial(self.weights, len(self.weights), replacement=True).numpy()

        self.allinds = []
        for cc in cinds:
            # print(cc)
            self.allinds.extend(np.where(np.isin(self.id_map, self.unique[cc]))[0].tolist())
            ccount += 1
            
            if ccount == self.bsize:
                # print(self.allinds)
                yield self.allinds
                self.allinds = []
                ccount = 0
        
        if len(self.allinds) > 0:
            yield self.allinds
            ccount = 0
            self.allinds= []

        # print(np.sum((np.diff(self.allinds)>1)))
        
        # return iter(self.allinds)

    def __len__(self):

        return len(self.allinds)

    def set_epoch(self, epoch):
        self.epoch = epoch


class distributed_balanced_time_coherent_sampler(Sampler):

    def __init__(self, dataset, nlags, bsize = 20, num_replicas=None, rank=None, shuffle=True):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()

        self.dataset = dataset
        # self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(np.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle

        self.filenames = self.dataset.filenames
       
        self.id_map = np.array([[int(x.split('_')[0].split('c')[-1]), x.split('_')[2]] for x in self.filenames])
        
        pattern = np.unique(self.id_map[:,1])
        
        self.unique, counts = np.unique(self.id_map, axis=0, return_counts = True)
        
        # unique is unique cell IDs, counts is number of instances of each cell ID
        self.weights = np.zeros(len(self.unique))
        
        for pi, pp in enumerate(pattern):
            # values[pi] = 1. / ''.join(self.filenames).count(pp)
            class_weight = 1. / np.sum(counts[self.unique[:,1] == pp] - nlags + 1)
            # tlags += np.sum(counts[self.unique[:,1] == pp] - nlags + 1)
            self.weights[self.unique[:,1] == pp] = class_weight
        
        self.unique = self.unique[:,0].astype(np.int32)
        self.id_map = self.id_map[:,0].astype(np.int32)
        self.weights = torch.as_tensor(self.weights)
        self.bsize = bsize

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
    
        indices = torch.multinomial(self.weights, len(self.weights), replacement=True, generator=g).tolist()

        # add extra samples to make it evenly divisible
        # indices += indices[:(self.total_size - len(indices))]
        # assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:len(indices):self.num_replicas]
        # assert len(indices) == self.num_samples

        ccount = 0

        self.allinds = []
        for cc in indices:
            # print(cc)
            self.allinds.extend(np.where(np.isin(self.id_map, self.unique[cc]))[0].tolist())
            ccount += 1
            
            if ccount == self.bsize:
                # print(self.allinds)
                yield self.allinds
                self.allinds = []
                ccount = 0
        
        if len(self.allinds) > 0:
            yield self.allinds
            ccount = 0
            self.allinds= []


    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch

def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    ax.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    print('tests')
    ax.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    ax.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    ax.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    # plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])

    ax.savefig('test_grads.png')

    return


class ConvNet:
    # Initialize the class
    def __init__(self, train_data, valid_data, lr):

        if torch.cuda.is_available() == True:
            self.dtype_double = torch.cuda.FloatTensor
            self.dtype_int = torch.cuda.LongTensor
            # self.device = torch.device("cuda:0") 
            # cudnn.benchmark = True
        else:
            self.dtype_double = torch.FloatTensor
            self.dtype_int = torch.LongTensor
            # self.device = torch.device("cpu")
        
        self.train_data = train_data
        # pass in data loader
        self.valid_data = valid_data

        self.net = mc.convNN().double() #.to(self.device) #type(self.dtype_double)

        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')

        self.lr = lr
    
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)

    def adjust_learning_rate(self, optimizer, epoch):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        self.lr = self.lr * (0.1 ** (epoch // 50))
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.lr
        return

    def train(self, epochs, total): #, bsize = 128):
        # print('enter training')
#         bsize = int(np.ceil(len(traindata) * batch_pct))
        # self.train_loader = DataLoader(self.train_data, batch_size=bsize, shuffle=True)
        
        accuracy = []
        # total = len(np.vstack([i[1].detach().numpy() for i in self.train_data]))      
        start_time = timeit.default_timer()

        for epoch in range(epochs):

            self.train_data.sampler.set_epoch(epoch)
            self.adjust_learning_rate(self.optimizer, epoch)

            # scounts = np.zeros(3)
            correct = 0.
            losslist = []       
            for it, (images, labels, ids) in enumerate(self.train_data):
                    
                self.optimizer.zero_grad()

                # pixel-wise standardization in each batch
                images = (images - images.mean(dim=0)) / (images.std(dim=0) + 1e-8)

                images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True)
                                
                outputs = self.net(images)

                loss = self.loss_fn(outputs, labels.squeeze().long())
                
                loss.backward()

                # plot_grad_flow(self.net.named_parameters())

                self.optimizer.step()
                losslist.append(loss.detach_())

                if it % 20 == 0:
                    print(losslist[-1])

                predicted = torch.max(outputs, dim=1)[1].unsqueeze(1)
                correct += (predicted.detach().long().eq(labels.detach_().long())).sum()

                del loss
            
            # print(losslist)
            size = float(dist.get_world_size())
            acc = correct / (total / size) # because total is the total length of the dataset and each rank sees 1/size. might be off slightly due to distributed sampler but it's fine
            elapsed = timeit.default_timer() - start_time
            
            # print(size)
            # print(torch.stack(losslist))
            # print(torch.stack(losslist).mean())

            summaries = torch.tensor([torch.stack(losslist).mean(), acc, float(epoch), elapsed], requires_grad=False).cuda()
            # print(summaries)
            # sys.exit()

            dist.reduce(summaries,0,op=dist.ReduceOp.SUM)
            # sys.exit()
            if dist.get_rank() == 0:
                summaries /= size
                print('avg loss %f' % summaries[0])
                print('accuracy %f' % (summaries[1]*100))
                print('epoch %f, time %f s' % (summaries[2], summaries[3]))
                        
            start_time = timeit.default_timer()

            if summaries[0] < 0.5:
                # del images
                # del labels
                # del outputs
                torch.cuda.ipc_collect()
                
                return summaries

        return summaries

    def test(self, valid_data):

        self.net.eval()

        with torch.no_grad():

            correct = 0.
            total = 0.
            loss = 0.
            
            total = np.vstack([i[1].detach().numpy() for i in valid_data])
            u, c = np.unique(total, return_counts=True)
            # print(u)
            # print(c)
            total = len(total)
            class_dim = len(u)            

            c_matrix = np.zeros((class_dim, class_dim))
            losslist = []
            
            for it, (images, labels, ids) in enumerate(valid_data):

                images = (images - images.mean(dim=0)) / (images.std(dim=0) + 1e-8)

                images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True)
                                
                outputs = self.net(images)
                loss = self.loss_fn(outputs, labels.squeeze().long())
                losslist.append(loss)

                _, predicted = torch.max(outputs, dim=1)

                labels = labels.cpu().detach().numpy()
                predicted = predicted.unsqueeze(1).cpu().detach().numpy()
                correct += (predicted == labels).sum()
                # print(type(correct))
                
                for ai in np.arange(class_dim):
                    for pi in np.arange(class_dim):
                        c_matrix[pi,ai] += np.sum([(predicted == ai) & (labels == pi)])

            c_balance = c / np.sum(c)
            acc = correct / total
            size = float(dist.get_world_size())

            summaries1 = torch.tensor([torch.stack(losslist).mean(), acc], requires_grad=False).cuda()
            summaries2 = torch.tensor(c_balance, requires_grad=False).cuda()
            summaries3 = torch.tensor(c_matrix, requires_grad=False).cuda()

            dist.reduce(summaries1,0,op=dist.ReduceOp.SUM)
            dist.reduce(summaries2,0,op=dist.ReduceOp.SUM)
            dist.reduce(summaries3,0,op=dist.ReduceOp.SUM)
            
            if dist.get_rank() == 0:
                summaries1 /= size
                summaries2 /= size
                # summaries3 /= size
                print(summaries1)
                print('class balance')
                print(summaries2)
                print('confusion matrix')
                print(summaries3)
                print('avg loss %f' % summaries1[0])
                print('accuracy %f' % (summaries1[1]*100))
                # print('epoch %f, time %f s' % (summaries[2], summaries[3]))

        return summaries1, summaries2, summaries3



class ConvPlusLSTM:
    # Initialize the class
    def __init__(self, train_data, valid_data, nlags, hidden_dim):

        if torch.cuda.is_available() == True:
            self.dtype_double = torch.cuda.FloatTensor
            self.dtype_int = torch.cuda.LongTensor
            # torch.cuda.set_device(6) 
            # cudnn.benchmark = True
        else:
            self.dtype_double = torch.FloatTensor
            self.dtype_int = torch.LongTensor
            self.device = torch.device("cpu")
        
        # print(self.dtype_double)
        self.train_data = train_data
        # pass in data loader
        self.valid_data = valid_data

        self.cnet = mc.convOnly().float() #.cuda() #type(self.dtype_double)
        
        # print(type(self.cnet.state_dict()))
        # print(list(self.cnet.state_dict().keys())[-1])
        last_dim = next(reversed(self.cnet.state_dict().values())).shape

        # print(last_dim[0])
        # sys.exit()

        self.x_dim = last_dim[0]
        self.nlags = nlags
        self.hidden_dim = hidden_dim

        # self.y_dim = 32
        # self.lstm = mc.LSTM(self.x_dim, self.y_dim, self.nlags, hidden_dim).cuda()

        self.lstm = mc.torchLSTM(self.x_dim, hidden_dim, nlags).float() #.cuda()

        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')

        self.lr = 5e-2
        # print([i.shape for i in self.lstm.parameters()])
        # print(len([i.shape for i in self.lstm.parameters()]))

        # sys.exit()
    
        self.combo_optimizer = (
            torch.optim.Adam(list(self.cnet.parameters()) + list(self.lstm.parameters()), lr = self.lr)
        )

    def adjust_learning_rate(self, optimizer, epoch):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        self.lr = self.lr * (0.1 ** (epoch // 200))
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.lr
        return

    def train(self, epochs, total): #, bsize = 128):
        
#         bsize = int(np.ceil(len(traindata) * batch_pct))
        # self.train_loader = DataLoader(self.train_data, batch_size=bsize, shuffle=True)
        
        accuracy = []
        # total = len(np.vstack([i[1].detach().numpy() for i in self.train_data]))
        start_time = timeit.default_timer()
        for epoch in range(epochs):

            self.train_data.batch_sampler.set_epoch(epoch)
            self.adjust_learning_rate(self.combo_optimizer, epoch)
            
            correct = 0.
            batch_lags = 0. 
            losslist = []

            for it, (images, labels, ids) in enumerate(self.train_data):
                
                unq, cnt = np.unique(ids[0], return_counts=True)
                batch_lags += np.sum(cnt - self.nlags+1)
                # print(batch_lags)

                self.combo_optimizer.zero_grad()

                # pixel-wise standardization in each batch
                images = ((images - images.mean(dim=0)) / (images.std(dim=0) + 1e-8)).float()

                images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True)
                
                # print('image')
                outputs = self.cnet(images)

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
                
                laglabels = torch.stack(laglabels)
                inputs = torch.stack(restack, dim = 1)
                LSTMoutputs = self.lstm(inputs)

                loss = self.loss_fn(LSTMoutputs, laglabels.squeeze().long())
                loss.backward()
                
                # if dist.get_rank() == 0:
                #     plot_grad_flow(self.cnet.named_parameters())

                #     plot_grad_flow(self.lstm.named_parameters())

                # plt.show()
                # sys.exit()

                self.combo_optimizer.step()
                losslist.append(loss.detach_())               

                if it % 20 == 0:
                    print(losslist[-1])

                predicted = torch.max(LSTMoutputs, dim=1)[1].unsqueeze(1)
                correct += (predicted.detach().long().eq(laglabels.detach_().long())).sum()

                del loss; del images; del labels;
                del LSTMoutputs
                del inputs
                del outputs
                del laglabels
                torch.cuda.ipc_collect()

            print('epoch batch lags: %f' % batch_lags)
            size = float(dist.get_world_size())
            acc = correct / batch_lags # because total is the total length of the dataset and each rank sees 1/size. might be off slightly due to distributed sampler but it's fine
            elapsed = timeit.default_timer() - start_time

            summaries = torch.tensor([torch.stack(losslist).mean(), acc, float(epoch), elapsed], requires_grad=False).cuda()

            dist.reduce(summaries,0,op=dist.ReduceOp.SUM)
            # sys.exit()
            if dist.get_rank() == 0:
                summaries /= size
                print('avg loss %f' % summaries[0])
                print('accuracy %f' % (summaries[1]*100))
                print('epoch %f, time %f s' % (summaries[2], summaries[3]))
                        
            start_time = timeit.default_timer()

            if summaries[0] < 0.5:
                # del images
                # del labels
                # del outputs
                torch.cuda.ipc_collect()
                
                return summaries

        return summaries

    def test(self, valid_data):

        self.cnet.eval()
        self.lstm.eval()

        with torch.no_grad():

            correct = 0.
            total = 0.
            loss = 0.
            
            total = np.vstack([i[1].detach().numpy() for i in valid_data])
            u, c = np.unique(total, return_counts=True)
            total = len(total)
            class_dim = len(u)

            c_matrix = np.zeros((class_dim, class_dim))
            losslist = []
            
            for it, (images, labels, ids) in enumerate(valid_data):
                
                unq, cnt = np.unique(ids[0], return_counts=True)
                batch_lags = np.sum(cnt - self.nlags+1)
                # pixel-wise standardization in each batch
                images = ((images - images.mean(dim=0)) / (images.std(dim=0) + 1e-8)).float()

                images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True)
                                
                outputs = self.cnet.forward(images)

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

                LSTMoutputs = self.lstm.forward(inputs)

                loss = self.loss_fn(LSTMoutputs, laglabels.squeeze().long())
                losslist.append(loss)
    #             print(loss)
                _, predicted = torch.max(LSTMoutputs, dim=1) # .detach().numpy()

                labels = laglabels.cpu().detach().numpy()
                predicted = predicted.unsqueeze(1).cpu().detach().numpy()
                correct += (predicted == labels).sum()
                # print(type(correct))
                
                for ai in np.arange(class_dim):
                    for pi in np.arange(class_dim):
                        c_matrix[pi,ai] += np.sum([(predicted == ai) & (labels == pi)])

            c_balance = c / np.sum(c)
            acc = correct / total
            size = float(dist.get_world_size())

            summaries1 = torch.tensor([torch.stack(losslist).mean(), acc], requires_grad=False).cuda()
            summaries2 = torch.tensor(c_balance, requires_grad=False).cuda()
            summaries3 = torch.tensor(c_matrix, requires_grad=False).cuda()

            dist.reduce(summaries1,0,op=dist.ReduceOp.SUM)
            dist.reduce(summaries2,0,op=dist.ReduceOp.SUM)
            dist.reduce(summaries3,0,op=dist.ReduceOp.SUM)
            
            if dist.get_rank() == 0:
                summaries1 /= size
                summaries2 /= size
                # summaries3 /= size
                print(summaries1)
                print('class balance')
                print(summaries2)
                print('confusion matrix')
                print(summaries3)
                print('avg loss %f' % summaries1[0])
                print('accuracy %f' % (summaries1[1]*100))
                # print('epoch %f, time %f s' % (summaries[2], summaries[3]))

        return summaries1, summaries2, summaries3

class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x



# if __name__ == "__main__":
	
#     pd.set_option('display.expand_frame_repr', False, 'display.max_columns', None)
#     cwd = os.getcwd()

#     if 'Chris Price' in cwd:
#         datadir = 'C:\\Users\\Chris Price\\Box Sync\\Plasticity_Protrusions_ML\\'
#     elif 'ccarr' in cwd:
#         datadir = 'C:\\Users\\ccarr_nj7afa9\\Box Sync\\Plasticity_Protrusions_ML\\'
#     else:
#         print('add your path to Plasticity_Protrusions_ML here')
#         sys.exit()

#     bsize = 256
# # ################################# random sampler
# #     cell_set = single_cell_dataset(datadir +'images\\data-images\\train_series.zip')
# #     test_randloader = DataLoader(cell_set, batch_size=bsize, shuffle=True)

#     train_set = single_cell_dataset(datadir +'images\\data-images\\train_series.zip')
#     test_set = single_cell_dataset(datadir +'images\\data-images\\test_series.zip')

#     train_randloader = DataLoader(train_set, batch_size=bsize, shuffle=True)
#     test_randloader = DataLoader(test_set, batch_size=bsize, shuffle=True)

#     # for it, (images, labels, ids) in enumerate(test_randloader):
#     #     if it < 1:
#     #         print(ids)
#     #     else:
#     #         break
# ####################################   

# # ############################## time sampler
# #     num_lags = 6

# #     cohsample = time_coherent_sampler(cell_set, batch_size = int(bsize/num_lags))
# #     # this is pretty garbage, but as long as batch_size here is large enough relative to batch_size in the sampler, you will get the complete time series returned for all the cells.
# #     # needs a better implementation with sampler, batch_sampler. or honestly just a batching class that's not a dataloader.
# #     test_cohloader = DataLoader(cell_set, sampler=cohsample, batch_size = 1000)

# #     for it, (images, labels, ids) in enumerate(test_cohloader):

# #         if it < 1:
# #             print(len(np.unique(ids[0])))
# #             print(len(ids[0]))
# #             print(np.unique(ids[0]))
# #             print(ids[1])
# #         else:
# #             break
# # ###################################
	
# ########### test basic CNN

#     cnetmodel = ConvNet(train_randloader, test_randloader) #, testdata)
#     lossL = cnetmodel.train(epochs = 5)
