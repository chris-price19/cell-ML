import numpy as np
import scipy
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
import time
from torch.autograd import Variable,grad
import pandas as pd
import random
import mlxtend
from mlxtend.preprocessing import standardize
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from numpy import array
from numpy.random import choice
data = pd.read_csv("C:\\Users\\as036\\.spyder-py3\\cellvector.csv")





df = pd.DataFrame(data, columns = ['time','x','y','z','trackID','spheric','ellipticO','ellipticP','area','volume','time_int','env','plastic','assay'])


assaymap = {22:1,26:2,27:3,28:4,29:5,30:6,33:7,34:8,35:9,36:10,37:11,38:12,41:13,43:14}
mapping = {'L':0, 'H':2, 'M':1}
drugmap = {'dH20':'dH20','dH2O':'dH20','IgG':'IgG','IgG ':'IgG','DMSO':'DMSO','DMSO ':'DMSO'}
setup9 = df.replace({'env':drugmap,'plastic':mapping,'assay':assaymap})
new_map = {'DMSO':0,'GM6001':1,'CK-666':2,'B1':3,'IgG':4,'NSC23766':5,'Y27632':6,'dH20':7,'Drug Z':8,'Bleb':9,'Lata':10,'Mar':11}
setup9 = setup9.replace({'env':new_map})
is_hol = setup9['plastic'] == 1
nextup = setup9['plastic'] == 0
set_try = setup9[is_hol]
set_up = setup9[nextup]
print(set_try['assay'].value_counts())
print(set_up['assay'].value_counts())
for i in range(4):
    setup9 = setup9.append(set_try)
for i in range(2):
    setup9 = setup9.append(set_up)


vectormaker = pd.DataFrame(setup9, columns = ['x','y','z','spheric','ellipticO','ellipticP','area','volume','env','plastic','assay','trackID','time_int'])
#reps = [3 if val == 0 else 1 for val in df]
standardize_this = ['x','y','z','spheric','ellipticO','ellipticP','area','volume']

vectormaker[standardize_this] = (vectormaker[standardize_this]-vectormaker[standardize_this].min())/(vectormaker[standardize_this].max()-vectormaker[standardize_this].min())
print(len(vectormaker['assay'].unique()))

checkplastic = pd.DataFrame(vectormaker, columns = ['plastic'])

checkenv = pd.DataFrame(vectormaker, columns = ['env'])


del setup9
del df
del data

print(vectormaker)


lags = 8





if torch.cuda.is_available() == True:
    print("SLINGSHOT ENGAGED")
else:
    print("If your not first you're last")






class myRNN(nn.Module):
    def __init__ (self, x ,  lags, layers,epochs):
        super(myRNN, self).__init__()
        if torch.cuda.is_available() == True:
            self.dtype = torch.cuda.FloatTensor
            self.device = 'cuda'
        else:
            self.dtype = torch.FloatTensor
            
            
        self.x = x
        self.layers = layers
        
        
        self.epochs = epochs
        self.xshape = list(self.dataset()[0][-1].size())[0]
        print(self.dataset()[1][-1].shape)
        self.yshape = self.dataset()[1][-1].shape
        
        #self.yshape = 
        self.hiddenshape = 20
        self.lags = lags
        self.random = np.random.randint(0,1332-self.lags)
       
        
        self.finalweights = Variable(torch.zeros(1,3).type(self.dtype),requires_grad=True)
        
        self.final_layer = Variable(self.gloroti(self.layers[1],3).type(self.dtype),requires_grad=True)
        
        #weight of forget gate h 
        self.w_fgateh = Variable(torch.eye(self.layers[1]).type(self.dtype),requires_grad=True)
        #weight of forget gate x
        self.w_fgatex = Variable(self.gloroti(self.layers[0],self.layers[1]).type(self.dtype),requires_grad=True)
        #bias of forget gate
        self.f_fgatex = Variable(self.gloroti(self.layers[0],self.layers[1]).type(self.dtype),requires_grad=True)
        self.g_fgatex = Variable(self.gloroti(self.layers[0],self.layers[1]).type(self.dtype),requires_grad=True)
        self.b_fgate=Variable(torch.zeros(1,self.layers[1]).type(self.dtype),requires_grad=True)
        #update gate weights h
        self.w_updateh = Variable(torch.eye(self.layers[1]).type(self.dtype),requires_grad=True)
        
        self.b_update = Variable(torch.zeros(1,self.layers[1]).type(self.dtype),requires_grad=True)
        # tanh gate weightsh
        self.w_tweighth = Variable(torch.eye(self.layers[1]).type(self.dtype),requires_grad=True)
        
        
        #tanh gate bias
        self.w_tbias = Variable(torch.zeros(1,self.layers[1]).type(self.dtype),requires_grad=True)
         # igate weightsh
        self.w_iweighth = Variable(torch.eye(self.layers[1]).type(self.dtype),requires_grad=True)
       
        #i gate bias
        self.w_ibias = Variable(torch.zeros(1,self.layers[1]).type(self.dtype),requires_grad=True)
        #kk
        self.optimizer = torch.optim.SGD([self.final_layer,
self.finalweights,
self.w_fgateh,
self.w_fgatex,
self.b_fgate,
self.w_updateh,
self.b_update,
self.w_tweighth,
self.w_tbias,
self.w_iweighth,
self.g_fgatex,
self.f_fgatex,
self.w_ibias],lr = 5e-5 )
        self.loss = nn.CrossEntropyLoss().cuda()
    def one_hotenc(self):
        data = ['DMSO','GM6001','CK-666','B1','IgG','NSC23766','Y27632','dH20','Drug Z','Bleb','Lata','Mar']
        values = array(data)
        print(values)
        label_encoder = LabelEncoder()
        integer_encoded = label_encoder.fit_transform(values)
        print(integer_encoded)
        onehot_encoder = OneHotEncoder(sparse = False)
        integer_encoded = integer_encoded.reshape(len(integer_encoded),1)
        onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
        print(onehot_encoded)
        

        print(one_hot)
    def gloroti(self, input_layer, output_layer):
        weights = np.random.normal(0,np.sqrt(2/(input_layer+output_layer)),input_layer)
        weights = np.repeat(weights, output_layer)
        weights = torch.tensor(weights, requires_grad=True).to(self.device)
        weights = weights.type(torch.cuda.FloatTensor).resize(input_layer,output_layer)
        return(weights) 
    def dataset(self):
        g = np.arange(0,12,1)
        t = np.arange(0,3,1)
        f = (3/4)/13
        l = (1/4)/13
        onehot = OneHotEncoder(sparse = False)
        twohot = OneHotEncoder(sparse = False)
        env_code = g.reshape(len(g),1)
        pls_code = t.reshape(len(t),1)
        onehot_env = onehot.fit_transform(env_code)
        onehot_pls = twohot.fit_transform(pls_code)
        

        minibatch = torch.zeros(lags, 24, 8)
        checkplas = torch.zeros(lags, 24, 1)
        checkenv = torch.zeros(lags, 24, 1)
        for j in range(24):
            i = random.choices(self.x['assay'].unique(),
            weights = [l,f,f,f,f,f,l,l,l,l,l,l,l,l],
            k = 1)
            setup1 = vectormaker['assay'] == float(i[0])
            setup1 = vectormaker[setup1]     
            p = random.choice(setup1['plastic'].unique())            
            setup2 = vectormaker['plastic'] == p
            setup2 = vectormaker[setup2]
            g = random.choice(setup2['env'].unique())
            setup3 = setup2['env'] == g
            setup3 = setup2[setup3]
            tre = random.choice(setup3['trackID'].unique() )
            setup4 = setup3['trackID'] == tre
            setup4 = setup3[setup4]
            while  setup4.shape[0]-(   lags+1) < 0:
                print('True')
                tre = random.choice(setup3['trackID'].unique())
                setup4 = setup3['trackID'] == tre
                setup4 = setup3[setup4]

            x = np.random.randint(0,setup4.shape[0]-(lags+1),1).item()
            
            setup5 = setup4.iloc[x:(x+lags),:]
            check_plas = pd.DataFrame(setup5, columns = ['plastic'])
            
            check_env = pd.DataFrame(setup5, columns = ['env'])
            
            setup5 = pd.DataFrame(setup5, columns = ['x','y','z','spheric','ellipticO','ellipticP','area','volume'])
            setup5 = torch.from_numpy(setup5.to_numpy())
            setup5 = setup5.type(torch.cuda.FloatTensor)
            
            check_plas = torch.from_numpy(check_plas.to_numpy())
            
            
            check_env = torch.from_numpy(check_env.to_numpy())

            check_plas = check_plas.type(torch.cuda.FloatTensor)
            check_env = check_env.type(torch.cuda.FloatTensor)
            minibatch[:,j,:] = setup5
            checkplas[:,j,:] = check_plas
            
            checkenv[:,j,:] = check_env
        
        
        integer_encoded = check_plas.reshape(len(check_plas),1)

        minibatch = minibatch.type(self.dtype)
        checkplas = checkplas[7,:,:]
        
        checkenv = checkenv[7,:,:]
        encode_env = checkenv.cpu().numpy()
        encode_env = onehot.transform(encode_env)
        
        encode_plas = checkplas.cpu().numpy()
        encode_plas = twohot.transform(encode_plas)
        
        checkplas = checkplas.type(self.dtype)
        checkenv = checkenv.type(self.dtype)
        
        return (minibatch, encode_plas, encode_env)
    def forwardpass(self, g1):
        
        h=torch.zeros(g1.shape[1],self.layers[1]).type(self.dtype)
        C_t = torch.zeros((g1.shape[1],self.layers[1])).type(self.dtype)
        
        for i in range(self.lags):
            f_t = torch.sigmoid(torch.matmul(h,self.w_fgateh)+torch.matmul(g1[i,:,:],self.w_fgatex)+self.b_fgate)
            i_t = torch.sigmoid(torch.matmul(h,self.w_iweighth)+torch.matmul(g1[i,:,:],self.g_fgatex)+self.w_ibias)
            tan_t = torch.tanh(torch.matmul(h,self.w_tweighth)+torch.matmul(g1[i,:,:],self.f_fgatex)+self.w_tbias)
            C_t = f_t*C_t+i_t*tan_t
            o_t = torch.sigmoid(torch.matmul(h,self.w_updateh)+torch.matmul(g1[i,:,:],self.w_fgatex)+self.b_update)
            h = o_t*(torch.tanh(C_t))
            
            
            
            
            
                
        
        h =  torch.matmul(h,self.final_layer)+self.finalweights
        
        
    
        h = torch.nn.Softmax(dim=1)(h)
        print("softmax probability",h[0].cpu().detach().numpy())
        return h
    def lossy(self, x,y):
        
        cent = 0
        for i in range(len(x)):
            for j in range(len(x[i])):
                
                
                cent += -1*y[i][j]*torch.log(x[i][j])
        return cent
    def predict(self, vector):
        vector = torch.from_numpy(vector).type(self.dtype)
        acc = self.forwardpass(vector)
        acc = acc.cpu().data.numpy()
        return acc
    def descendagradient(self):
        j = 0
        for i in range(self.epochs):
            loader = (self.dataset())
            print("low plasticity" , loader[1][:,0].sum())
            print("")
            print("medium plasticity" , loader[1][:,1].sum())
            print("")
            print("high plasticity", loader[1][:,2].sum())
            print("")
            inpt = self.forwardpass(loader[0])
            
            
            loss = self.lossy(inpt,loader[1])    
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            j += 1
            if i % 1 == 0:
                print("Loss",loss.cpu().item(), "at iteration", j)
    '''        
    def accuracycheck(self, check, checkacc):
        my_check = torch.zeros(24,1)
        for i in range(len(check)):
            for j in checkacc:
                my_check[i] = check[i]-j
                for j =
    '''          
rnn = myRNN(vectormaker,8,[8,50],10000)


rnn.forwardpass(rnn.dataset()[0])
print(rnn.dataset()[1][1])
rnn.descendagradient()
