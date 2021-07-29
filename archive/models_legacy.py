import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as utils
from torch.utils.tensorboard import SummaryWriter
import os
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn import metrics
from os.path import join
import numpy as np 
import time
from tqdm import tqdm


LARGEST_BATCH_SIZE=4096
SEED=234
torch.manual_seed(SEED)

class Softmax(nn.Module):
    def __init__(self,input_dim,num_classes):
        print("Initializing softmax")
        super(Softmax,self).__init__()
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self,x):
        output= self.classifier(x)         
        return output


class CNN2(nn.Module):
    # reference architecture:  https://github.com/vinayakumarr/Network-Intrusion-Detection/blob/master/UNSW-NB15/CNN/multiclass/cnn2.py
    def __init__(self,input_dim,num_classes):
        super(CNN2, self).__init__()
        # kernel
        print("Initializing CNN")
        self.input_dim = input_dim
        self.num_classes = num_classes

        conv_layers = []
        conv_layers.append(nn.Conv1d(in_channels=1,out_channels=64,kernel_size=3,padding=1)) # ;input_dim,64
        conv_layers.append(nn.ReLU(True))

        conv_layers.append(nn.Conv1d(in_channels=64,out_channels=64,kernel_size=3,padding=1)) #(input_dim,64)
        conv_layers.append(nn.ReLU(True))

        assert input_dim%2==0,"input dim has to be divisable by 2"
        conv_layers.append(nn.MaxPool1d(2)) # (input_dim/2,64)
        self.conv = nn.Sequential(*conv_layers)

        fc_layers = []

        fc_layers.append(nn.Linear(input_dim//2*64,256))
        fc_layers.append(nn.Dropout(p=0.5))
        fc_layers.append(nn.Linear(256,num_classes))
        self.classifier = nn.Sequential(*fc_layers)

    def forward(self, x):
        batch_size, D = x.shape
        #print("batch_siz,D",batch_size,D)
        x = x.view(batch_size,1,D)
        #print('x.shape = ',x.shape)
        x = self.conv(x)
        x = torch.flatten(x,1)
        #print("after flatten x.shape = ",x.shape)
        x = self.classifier(x)
        #print('after classifier x.shape = ',x.shape)
        return x


def get_sampler(weight,y):
    samples_weight = np.array([weight[t] for t in y])

    samples_weight = torch.from_numpy(samples_weight)
    samples_weight = samples_weight.double()
    sampler = torch.utils.data.sampler.WeightedRandomSampler(samples_weight, len(samples_weight))
    return sampler


class Classifier:
    def __init__(self,args,method):
        self.batch_size = int(args['batch_size'])
        self.num_iters = int(args['num_iters'])
        self.learning_rate = float(args['lr'])
        self.reg= float(args['reg'])
        self.runs_dir = args['runs_dir']
        self.device = torch.device(args['device'])
        self.balance = args['balance']
        self.validate_train = False
        self.class_weights = args['class_weight']
    
        num_class = int(args['num_class'])
        input_dim = int(args['input_dim'])
        if method=='softmax':
            model = Softmax(input_dim,num_class)
            self.model = model.to(self.device)
        elif method=='cnn2':
            model = CNN2(input_dim, num_class)
            self.model = model.to(self.device)
        
        balance = self.balance
        class_weights = self.class_weights
        if (balance=='with_loss' or balance=='sample_per_batch') and class_weights.any()==None:
            print("Non zero class weights required for ", balance)
            exit()
        
        if balance=='with_loss':
            self.criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights).to(device))
        elif balance=='sample_per_batch' or balance=='explicit' or balance=='no':
            self.criterion = nn.CrossEntropyLoss()
        else:
            print("There is #{}# balancing technique implemented ".format(balance))
            exit()

        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=self.learning_rate,betas=(0.9,0.99),eps=1e-08, weight_decay=self.reg, amsgrad=False)


    def fit(self,X,Y):
        print("start training model")
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=SEED)
        for dev_index, val_index in sss.split(X, Y): # runs only once
                X_dev = X[dev_index]
                Y_dev = Y[dev_index]
                X_val = X[val_index]
                Y_val = Y[val_index]  
        
        device=self.device
        
        tensor_x = torch.stack([torch.Tensor(i) for i in X_dev]).to(device)
        tensor_y = torch.LongTensor(Y_dev).to(device) # checked working correctly
	
        dataset = utils.TensorDataset(tensor_x,tensor_y)
        balance = self.balance
        if balance=='sample_per_batch':
            print("using weighted sampler with class weights")
            print(self.class_weights)
            sampler = get_sampler(self.class_weights,tensor_y) #to adress data imbalance.
            train_loader = utils.DataLoader(dataset,batch_size=self.batch_size,sampler=sampler)
        elif balance=='with_loss' or balance=='explicit' or balance=='no':
            train_loader = utils.DataLoader(dataset,batch_size=self.batch_size)
            #imbalance is already taken into account when creating architecture by passing class weights to loss function or explicitly over/under sampling
        
        N = tensor_x.shape[0]
        num_epochs = int(self.num_iters/(N/self.batch_size))+1
    
        model  = self.model.to(device)
        best_acc = None
        best_epoch = None

        filepath = join(self.runs_dir,'checkpoint.pth')
        if os.path.isfile(filepath):
            checkpoint = self.load_checkpoint(filepath)
            best_epoch = checkpoint['epoch']
            best_batch = checkpoint['batch']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            if 'training_complete' in checkpoint:
                if checkpoint['training_complete']=='yes':  # most likely all epochs are completed, not early stopping
                    print("training for this model is already copmleted with {}/{} epochs!".format(best_epoch, num_epochs))
                    #return 

            pred = self.predict(X_val)
            best_acc = metrics.balanced_accuracy_score(Y_val,pred)*100
            resume_epoch = best_epoch  
            resume_batch = best_batch+1
        else:
            resume_epoch = 0
            resume_batch = 0
            best_acc = -1
            best_epoch = 0

        no_improvement = 0
        model_best_state = None
        print("best epoch {}, best batch {}".format(resume_epoch,resume_batch))
        print("bst acc ", best_acc)
        validation_interval_iters = (50*LARGEST_BATCH_SIZE)//self.batch_size # fixes issues with varying batch sizes
        
        writer = SummaryWriter(self.runs_dir) 
        
        for epoch in range(resume_epoch,num_epochs):
            print("epoch:", epoch)
            tick = time.time()
            
            if resume_epoch==epoch:
                start_batch=resume_batch
            else:
                start_batch = 0
            for i,(xi,yi) in enumerate(train_loader):
                if i<start_batch:
                    continue
                if i==0:
                    print("Batch size = ", xi.shape)
                outputs = model(xi)
                loss = self.criterion(outputs,yi)
                #print('loss.item() = ',loss.item())
                seen_so_far = self.batch_size*epoch*len(train_loader)+self.batch_size*(i+1) # fixes issues with different batch size
                writer.add_scalar('Loss/train',loss.item(),seen_so_far)
                #batckward, optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                if i%validation_interval_iters==0:
                    #print("Validating")
                    pred = self.predict(X_val)
                    balanced_acc = metrics.balanced_accuracy_score(Y_val,pred)*100
                    if balanced_acc > best_acc:
                        best_acc = balanced_acc
                        best_epoch = epoch
                        checkpoint = {
                        'state_dict': model.state_dict(),
                        'optimizer' : self.optimizer.state_dict(),
                        'epoch':epoch,
                        'batch': i,
                        'batch_size': self.batch_size,
                        'training_complete':'no'
                        }
                        self.save(checkpoint)

                        #for saving model after training completes
                        model_best_state = model
                        optimizer_best_state = self.optimizer
                        best_epoch = epoch
                        best_batch = i
                                            
                    writer.add_scalar('Accuracy/Balanced Val',balanced_acc,seen_so_far)

                    acc = metrics.accuracy_score(Y_val,pred)*100
                    writer.add_scalar('Accuracy/Val',acc,seen_so_far)
                    
                    #print("Calc. train score.. ")
                    if self.validate_train==True:
                        train_pred = self.predict(X_dev)
                        train_acc = 100*metrics.accuracy_score(train_pred,Y_dev)
                        train_balanced_acc = 100*metrics.balanced_accuracy_score(train_pred,Y_dev)
                        writer.add_scalar('Accuracy/Train',train_acc,seen_so_far)
                        writer.add_scalar('Accuracy/Balance Train',train_balanced_acc,seen_so_far)

                    print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Balanced Acc: {:.2f}' 
                                               .format(epoch+1, num_epochs, i+1, len(Y_dev)//self.batch_size, loss.item(),balanced_acc))
            tock = time.time()
            t=int(tock-tick)
            print("Epoch completed in {:.0f} min ".format(t//60))
            if epoch>(best_epoch)+2:
                print("early stopping, no improvement  for two epoch ")
                break 

        writer.close()
        # saving complete model
        if model_best_state != None:
            checkpoint = {
                        'state_dict': model_best_state.state_dict(),
                        'optimizer' : optimizer_best_state.state_dict(),
                        'epoch':best_epoch,
                        'batch': best_batch,
                        'batch_size': self.batch_size,
                        'training_complete':'yes'
                        }
        self.save(checkpoint)


    def predict(self,x,bs=LARGEST_BATCH_SIZE,eval_mode=False,device=None):
        if device==None:
            device = self.device
        tensor_x = torch.stack([torch.Tensor(i) for i in x]).to(device)
        if bs is None:
            bs = self.batch_size
        num_batch = x.shape[0]//bs +1*(x.shape[0]%bs!=0)

        pred = torch.zeros(0,dtype=torch.int64).to(device)
        
        if eval_mode:
            model = self.load_model()
        else:
            model = self.model
        model = model.to(device)
        model.eval()        
        
        with torch.no_grad():
            for i in range(num_batch):
                xi = tensor_x[i*bs:(i+1)*bs]
                outputs = model(xi)
                _, predi = torch.max(outputs.data,1)
                pred = torch.cat((pred,predi))

        return pred.cpu().numpy()


    def save(self,checkpoint):
        path = join(self.runs_dir,'checkpoint.pth')
        torch.save(checkpoint,path)

    
    def load_checkpoint(self,filepath):
        if os.path.isfile(filepath):
            #print("Calling torch load")
            checkpoint = torch.load(filepath,map_location=self.device )
            #print("Checkpoint loaded")
            #print("Loaded {} model trained with batch_size = {}, seen {} epochs and {} mini batches".
            #    format(self.runs_dir,checkpoint['batch_size'],checkpoint['epoch'],checkpoint['batch'])) 
            return checkpoint
        else:
            print("No checkpoint foound for ",filepath)
            return None
        
            
    def load_model(self,inference_mode=True):
        filepath = join(self.runs_dir,'checkpoint.pth')
        #print("Loading checkpoint from ",filepath)
        checkpoint = self.load_checkpoint(filepath)
        
        model = self.model
        state_dict = checkpoint['state_dict']
        for k in state_dict.keys():
            if 'module.' in k:
                wrapped_by_data_parallel = True
            else:
                wrapped_by_data_parallel = False
            break
        if wrapped_by_data_parallel:
            state_dict= {k.partition('module.')[2]:state_dict[k] for k in state_dict.keys() }
        model.load_state_dict(state_dict)
        
        if inference_mode:
            for parameter in model.parameters():
                parameter.requires_grad = False
            model.eval()
        return model

