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
import matplotlib.pyplot as plt

#select folder to which files are saved
data = pd.read_csv("C:\\Users\\as036\\.spyder-py3\\cellvector2.csv")




#pick specific data that one might use in training
df = pd.DataFrame(data, columns = ['time','x','y','z','stepsize','trackID','spheric','ellipticO','ellipticP','area','volume','time_int','env','plastic','assay'])

#the following are simply interger maps and file cleaning for ease of data use
assaymap = {22:1,26:2,27:3,28:4,29:5,30:6,33:7,34:8,35:9,36:10,37:11,38:12,41:13,43:14}
mapping = {'L':0, 'H':2, 'M':1}
drugmap = {'dH20':'dH20','dH2O':'dH20','IgG':'IgG','IgG ':'IgG','DMSO':'DMSO','DMSO ':'DMSO'}
setup9 = df.replace({'env':drugmap,'plastic':mapping,'assay':assaymap})
new_map = {'DMSO':0,'GM6001':1,'CK-666':2,'B1':3,'IgG':4,'NSC23766':5,'Y27632':6,'dH20':7,'Drug Z':8,'Bleb':9,'Lata':10,'Mar':11}
setup9 = setup9.replace({'env':new_map})


#Picking out, in this case drug testing, setting to enviroments which are standardized
is_hol = setup9[(setup9['plastic'] == 2)]
is_hol = is_hol[(is_hol['time_int'] == 20.0)]

#print(is_hol['assay'].unique())
'''
time.sleep(10)
set_try = setup9[is_hol]
set_up = set_try[nextup]
finalup = set_up['plastic'] == 2
set_up = set_up[finalup]
print(set_up['assay'].unique())


print(set_try['assay'].value_counts())
print(set_up['assay'].value_counts())
for i in range(4):
    setup9 = setup9.append(set_try)
for i in range(2):
    setup9 = setup9.append(set_up)
'''

#Standardizing and data orginization
vectormaker = pd.DataFrame(is_hol, columns = ['stepsize','spheric','ellipticO','ellipticP','area','volume','env','plastic','assay','trackID','time_int'])
#reps = [3 if val == 0 else 1 for val in df]
standardize_this = ['stepsize','spheric','ellipticO','ellipticP','area','volume']

vectormaker[standardize_this] = (vectormaker[standardize_this]-vectormaker[standardize_this].min())/(vectormaker[standardize_this].max()-vectormaker[standardize_this].min())
print(len(vectormaker['assay'].unique()))

checkplastic = pd.DataFrame(vectormaker, columns = ['plastic'])

checkenv = pd.DataFrame(vectormaker, columns = ['env'])


del setup9
del df
del data

print(vectormaker['assay'].unique())






#Checking the gpu
if torch.cuda.is_available() == True:
    print("SLINGSHOT ENGAGED")
else:
    print("If your not first you're last")





#Module Runs entire network
class myRNN(nn.Module):
    def __init__ (self, x ,  lags, layers,conv_layers, epochs):
        super(myRNN, self).__init__()
        if torch.cuda.is_available() == True:
            self.dtype = torch.cuda.FloatTensor
            self.device = 'cuda'
            
        else:
            self.dtype = torch.FloatTensor
            
            
        self.x = x
        self.layers = layers
        self.lags= lags
        #Must be array denotes convolutional layer size, 4 layers currently
        self.lay = conv_layers
        
        self.epochs = epochs
        self.xshape = list(self.dataset()[0][-1].size())[0]
        print(self.xshape)
        print(self.dataset()[1][-1].shape)
        self.yshape = self.dataset()[1][-1].shape
        
        #self.yshape = 
        self.hiddenshape = 20
        self.lags = lags
        self.random = np.random.randint(0,1332-self.lags)
       
        #second number here must be the dimension of the output, in softmax terms true for self.finalweights, self.final_layer
        self.finalweights = Variable(torch.zeros(1,25).type(self.dtype),requires_grad=True)
        
        self.final_layer = Variable(self.gloroti(self.layers[1],25).type(self.dtype),requires_grad=True)
        
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
        #Con layers 
        self.Conv1 = nn.Conv2d(1, self.lay[0], kernel_size=(1,3), stride = 1).cuda()
        self.Conv2 = nn.Conv2d(self.lay[0], self.lay[1], kernel_size=(1,2), stride = 1).cuda()
        self.Conv3 = nn.Conv2d(self.lay[1], self.lay[2], kernel_size=(1,2), stride = 1).cuda()
        self.Conv4 = nn.Conv2d(self.lay[2], self.lay[3], kernel_size=(1,2), stride = 1).cuda()
        self.maxpool = nn.MaxPool2d(2, stride = 2).cuda()
        self.linearlayer = nn.Linear(25,25).cuda()
        self.finallayer = nn.Linear(25,11).cuda()
        
        self.params = list(self.Conv1.parameters()) + list(self.Conv2.parameters()) + list(self.Conv3.parameters()) + list(self.Conv4.parameters())+list(self.linearlayer.parameters())+list(self.finallayer.parameters())
        #Optimize networks seperatly
        self.optimizeruno = torch.optim.Adam(self.params,lr=2.5e-4)
        self.optimizer = torch.optim.Adam([self.final_layer,
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
self.w_ibias,
],lr = 2.5e-3 )
        self.loss = nn.CrossEntropyLoss().cuda()
    #Glorot initilization for all linear layer
    def gloroti(self, input_layer, output_layer):
        weights = np.random.normal(0,np.sqrt(2/(input_layer+output_layer)),input_layer)
        weights = np.repeat(weights, output_layer)
        weights = torch.tensor(weights, requires_grad=True).to(self.device)
        weights = weights.type(torch.cuda.FloatTensor).resize(input_layer,output_layer)
        return(weights) 
    #Takes batchs from our dataset, sets them up
    def dataset(self):
        #One hot encoder
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
        

        minibatch = torch.zeros(24,1,self.lags, 6)
        checkplas = torch.zeros(self.lags, 24, 1)
        checkenv = torch.zeros(self.lags, 24, 1)
        for j in range(24):
            #Picking assays, using weights for some balancing
            i = random.choices(self.x['assay'].unique(),
            weights = [f,f,f,f,f,f,f,f],
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
            while  setup4.shape[0]-(self.lags+1) < 0:
                it = 0
                tre = random.choice(setup3['trackID'].unique())
                setup4 = setup3['trackID'] == tre
                setup4 = setup3[setup4]
                it += 1
                if it == 500:
                    print("infinite while loop")
                    

            x = np.random.randint(0,setup4.shape[0]-(self.lags),1).item()
            
            setup5 = setup4.iloc[x:(x+self.lags),:]
            
            check_plas = pd.DataFrame(setup5, columns = ['plastic'])
            
            check_env = pd.DataFrame(setup5, columns = ['env'])
            #this line must match whatever data you are training against. 
            setup5 = pd.DataFrame(setup5, columns = ['stepsize','spheric','ellipticO','ellipticP','area','volume'])
            setup5 = torch.from_numpy(setup5.to_numpy())
            setup5 = setup5.type(torch.cuda.FloatTensor)
            
            check_plas = torch.from_numpy(check_plas.to_numpy())
            
            
            check_env = torch.from_numpy(check_env.to_numpy())

            check_plas = check_plas.type(torch.cuda.FloatTensor)
            
            check_env = check_env.type(torch.cuda.FloatTensor)
            minibatch[j,:,:,:] = setup5.unsqueeze(dim=0)
            checkplas[:,j,:] = check_plas
            
            checkenv[:,j,:] = check_env
        
        
        integer_encoded = check_plas.reshape(len(check_plas),1)

        minibatch = minibatch.type(self.dtype)
        checkplas = checkplas[self.lags-1,:,:]
        
        checkenv = checkenv[self.lags-1,:,:]
        encode_env = checkenv.cpu().numpy()
        encode_env = onehot.transform(encode_env)
        
        encode_plas = checkplas.cpu().numpy()
        encode_plas = twohot.transform(encode_plas)
        
        checkplas = checkplas.type(self.dtype)
        checkenv = checkenv.type(self.dtype)
        
        return (minibatch, encode_plas, encode_env)
        #Standard operating procedure, CNN-LSTM NET
    def forwardpass(self, g1):
        
        h=torch.zeros(g1.shape[0],self.layers[1]).type(self.dtype)
        C_t = torch.zeros((g1.shape[1],self.layers[1])).type(self.dtype)
        
        g1 = F.relu(self.Conv1(g1))
        
        g1 = F.relu(self.Conv2(g1))
        
        g1= F.relu(self.Conv3(g1))
        
        g1 = F.relu(self.Conv4(g1))
        
        g1 = self.maxpool(g1)
        
        g1 = g1.squeeze(dim=3)
        
                
        
        
        for i in range(len(g1[1])):
            
            f_t = torch.sigmoid(torch.matmul(h,self.w_fgateh)+torch.matmul(g1[:,i,:],self.w_fgatex)+self.b_fgate)
            
            i_t = torch.sigmoid(torch.matmul(h,self.w_iweighth)+torch.matmul(g1[:,i,:],self.g_fgatex)+self.w_ibias)
            tan_t = torch.tanh(torch.matmul(h,self.w_tweighth)+torch.matmul(g1[:,i,:],self.f_fgatex)+self.w_tbias)
            C_t = f_t*C_t+i_t*tan_t
            o_t = torch.sigmoid(torch.matmul(h,self.w_updateh)+torch.matmul(g1[:,i,:],self.w_fgatex)+self.b_update)
            h = o_t*(torch.tanh(C_t))
            
            
            
            
            
        
        
            
            
                
        
        h =  torch.matmul(h,self.final_layer)+self.finalweights
        
        h = torch.tanh(self.linearlayer(h))
        h = self.finallayer(h)
        
        h = torch.nn.Softmax(dim=1)(h)
        
        return h

    def lossy(self, x,y):
        
        cent = 0
        for i in range(len(x)):
            for j in range(len(x[i])):
                
                #Cross entropy loss, output of layers is a softmax
                cent += -1*y[i][j]*torch.log(x[i][j])
        return cent
    def predict(self, vector):
        vector = torch.from_numpy(vector).type(self.dtype)
        acc = self.forwardpass(vector)
        acc = acc.cpu().data.numpy()
        return acc
    def descendagradient(self):
        j = 0
        track = np.zeros((int(self.epochs/200),1))
        graph = np.zeros((int(self.epochs/200),1))
        bins = np.arange(0,int(self.epochs/200),1)
        for i in range(self.epochs):
            loader = (self.dataset())
            checkem = (self.dataset())
            inpt = self.forwardpass(loader[0])
            
            
            loss = self.lossy(inpt,loader[2])    
            
            loss.backward()
            self.optimizer.step()
            self.optimizeruno.step()
            self.optimizer.zero_grad()
            self.optimizeruno.zero_grad()
            j += 1
            if (i+1)% 200 == 0:
                print("Loss",loss.cpu().item(), "at iteration", j)
                print("low plasticity" , checkem[1][:,0].sum())
                print("")
                print("medium plasticity" , checkem[1][:,1].sum())
                print("")
                print("high plasticity", checkem[1][:,2].sum())
                print("")
                print(inpt[0].detach().cpu().numpy())
                print( "Current Accuracy" , self.accuracycheck(checkem[0], checkem[2])[0])
                print("Confusion Matrix")
                track[int((i+1)/200-1)]= (loss.detach().cpu().numpy())/24
                print(self.accuracycheck(checkem[0], checkem[2])[2])
                graph[int((i+1)/200-1)] = self.accuracycheck(checkem[0], checkem[2])[0]
            
        return graph, bins, track

    def accuracycheck(self, x, y ):
        loader = self.forwardpass(x).detach().cpu().numpy()
        checker = np.zeros((loader.shape))
        idx = np.zeros((len(loader)))
        summable = 0
        confused = np.zeros((len(loader[0])+1,len(loader[0])+1))
        for i in range(len(loader)):
            idx = np.argmax(loader[i])
            pdx = np.argmax(y[i])
            checker[i, idx] = 1  
            confused[idx][pdx] += 1
            if y[i, idx] == 1:
                summable += 1
        
                
            
        
        summable = summable/24

        
        
        return summable, checker, confused

         
            

        
# first layer must be the dimensionality of the data, 
#first input is data, second is lags, third is one neural layer, last is epochs
rnn = myRNN(vectormaker,16  ,[8,80],[16,16,32,32],15000)




graph, bins, track = rnn.descendagradient()
fig, ax = plt.subplots()

jones = ax.plot(bins, graph, '-r', color ='green', label = 'training accuracy', markersize = 0.5)
mike = ax.plot(bins, track, '-r', color = 'blue', label = 'loss function')
plt.title('Accuracy vs Training')
plt.savefig('C:\\Users\\as036\\OneDrive\\Documents\\cancer_figures' + 'myCNNLSTM graph.png', markersize = 0.5)