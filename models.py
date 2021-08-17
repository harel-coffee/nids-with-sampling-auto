import torch
import torch.nn as nn
import torch.utils.data as data
import torch.utils.data as utils
from torch.utils.tensorboard import SummaryWriter

import os
from sklearn import metrics
from os.path import join
import numpy as np 
import time
from tqdm import tqdm
import pandas as pd

from dataset_loader import FlowRecordDataset
from utils import get_cols4ml, encode_label
from utils import getSeed
import ntpath


LARGEST_BATCH_SIZE=4096
torch.manual_seed(getSeed())

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
        x = x.view(batch_size,1,D)
        x = self.conv(x)
        x = torch.flatten(x,1)
        x = self.classifier(x)
        return x


def get_sampler(weight,y):
    samples_weight = np.array([weight[t] for t in y])

    samples_weight = torch.from_numpy(samples_weight)
    samples_weight = samples_weight.double()
    sampler = torch.utils.data.sampler.WeightedRandomSampler(samples_weight, len(samples_weight))
    return sampler


class Classifier:
    def __init__(self,args,method):
        self.batch_size = args['batch_size']
        self.learning_rate = args['lr']
        self.reg= args['reg']
        
        self.runs_dir = args['runs_dir']
        self.balance = args['balance']
        self.class_weights = args['class_weight']
        self.device = torch.device(0)
        self.classes_ = np.arange(args['num_class']) 
        num_class = args['num_class']
        input_dim = args['input_dim']
        if method=='softmax':
            model = Softmax(input_dim,num_class)
        elif method=='cnn2':
            model = CNN2(input_dim, num_class)
        self.model = model.to(self.device)
        
        balance = self.balance
        print("balancing technique ", balance)
        class_weights = self.class_weights
        if balance=='with_loss' or balance=='sample_per_batch' or balance=='with_loss_inverse':
            assert class_weights.any()!=None, "Non zero class weights required for {} balancing".format( balance)
        
        if balance=='with_loss' or balance=='with_loss_inverse':
            self.criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights).to(self.device))
        elif balance=='sample_per_batch' or balance=='explicit' or balance=='no':
            self.criterion = nn.CrossEntropyLoss()
        else:
            print("There is #{}# balancing technique implemented ".format(balance))
            exit()

        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=self.learning_rate,betas=(0.9,0.99),eps=1e-08, weight_decay=self.reg, amsgrad=False)


    def fit(self, csv_files, csv_file_val, num_epochs, resume=True):
        print("start training model", ntpath.basename(self.runs_dir))
       
        device=self.device

        # load multiple datasets        
        list_of_datasets = []
        for csv_file in csv_files:
            list_of_datasets.append(FlowRecordDataset(csv_file))
        multipleRecordDataset = data.ConcatDataset(list_of_datasets) #https://stackoverflow.com/questions/53477861/pytorch-dataloader-multiple-data-source

        balance = self.balance
        if balance=='sample_per_batch':
            print("using weighted sampler with class weights")
            print(self.class_weights)
            sampler = get_sampler(self.class_weights,tensor_y) #to adress data imbalance.
            train_loader = utils.DataLoader(multipleRecordDataset,batch_size=self.batch_size,sampler=sampler)
        elif balance=='with_loss' or balance=='with_loss_inverse' or balance=='explicit' or balance=='no':
            train_loader = utils.DataLoader(multipleRecordDataset,batch_size=self.batch_size)
            #imbalance is already taken into account when creating architecture by passing class weights to loss function or explicitly over/under sampling
        print("Initializing val dataset")
        val_dataset = FlowRecordDataset(csv_file_val)

        model  = self.model.to(device)
        best_acc = 0
        best_epoch = None

        filepath = join(self.runs_dir,'checkpoint.pth')
        if os.path.isfile(filepath):
            checkpoint = self.load_checkpoint(filepath)
            best_epoch = checkpoint['epoch']
            best_batch = checkpoint['batch']
            self.model.load_state_dict(checkpoint['state_dict'])
            model = self.model
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            if 'training_complete' in checkpoint:# do not execute temporarily, debugging 
                if checkpoint['training_complete']=='yes':  # most likely all epochs are completed, not early stopping
                    print("training for this model is already copmleted with {}/{} epochs!".format(best_epoch, num_epochs))
                    return
            else:
                print('Resuming from epoch ', best_epoch, 'batch {}/{}'.format(best_batch,len(train_loader)))
            print('get_best_acc')
            best_acc, acc = self.get_val_accuracy(model, val_dataset)
            model.train()
            torch.set_grad_enabled(True)
           
            resume_epoch = best_epoch
            resume_batch = best_batch+1
        else:
            resume_epoch = 0
            resume_batch = 0
            best_acc = -1
            best_epoch = 0

        no_improvement = 0
        model_best_state = None
        validation_interval_iters = (100*LARGEST_BATCH_SIZE)//self.batch_size # fixes issues with varying batch sizes
        print("Validation interval: ",validation_interval_iters)
 
        writer = SummaryWriter(self.runs_dir) 
        
        for epoch in range(resume_epoch,num_epochs):
            train_loader = utils.DataLoader(multipleRecordDataset,batch_size=self.batch_size)
            print("epoch:", epoch)
            epoch_started = time.time()
            
            if resume_epoch==epoch:
                start_batch=resume_batch
            else:
                start_batch=0
            for i,batch in enumerate(train_loader):
                if i<start_batch:
                    continue# cuz already trained that part
                xi = batch[0].to(device)
                yi = batch[1].to(device)
                outputs = model(xi)
                loss = self.criterion(outputs,yi)
                seen_so_far = self.batch_size*epoch*len(train_loader)+self.batch_size*(i+1) # fixes issues with different batch size
                writer.add_scalar('Loss/train',loss.item(),seen_so_far)
                #batckward, optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                if i%validation_interval_iters==0:
                    #print("Validating")
                    tick = time.time()
                    balanced_acc, acc = self.get_val_accuracy(model, val_dataset)
                    if i==0:
                        print("Time for validation: {:.2f} sec".format(time.time()-tick))
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
                    writer.add_scalar('Accuracy/Val',acc,seen_so_far)
                    
                    _, pred = torch.max(outputs.data,1)
                    train_acc = metrics.balanced_accuracy_score(yi.cpu(),pred.cpu())*100
                    print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Balanced train/Val Acc: {:.2f}/{:.2f}' 
                                               .format(epoch+1, num_epochs, i+1, len(train_loader), loss.item(), train_acc, balanced_acc))
                    # get back to train mode
                    model.train()
                    torch.set_grad_enabled(True)
            tock = time.time()
            t=int(tock-epoch_started)
            print("Epoch completed in {:.0f} min ".format(t//60))
            if epoch>(best_epoch)+1:
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



    def get_val_accuracy(self,model, val_dataset):
        loader = utils.DataLoader(val_dataset, batch_size=LARGEST_BATCH_SIZE)
        torch.set_grad_enabled(False)
        model.eval()       
        y = torch.cat([y for i, (x,y) in enumerate(loader)], dim=0)
        outputs = torch.cat([model(x.to(self.device)) for i, (x,y) in enumerate(loader)], dim=0)
        _, pred = torch.max(outputs.data,1)
        
        balanced_acc  = metrics.balanced_accuracy_score(y,pred.cpu())*100
        acc = metrics.accuracy_score(y,pred.cpu())*100
        return balanced_acc, acc

    def predict_proba(self,x,bs=LARGEST_BATCH_SIZE,model=None,device=None):
        if device==None:
            device = self.device
        
        num_batch = x.shape[0]//bs +1*(x.shape[0]%bs!=0)
        proba = torch.zeros((0,self.model.num_classes),dtype=torch.int64).to(device)
         
        if model!=None:
            model = model
        else:
            model = self.load_model()
        model = model.to(device)
        
        torch.set_grad_enabled(False)
        model.eval()        
        softmax = nn.Softmax(dim=1)
        with torch.no_grad():
            for i in range(num_batch):
                xi = x[i*bs:(i+1)*bs]
                xi_tensor = torch.stack([torch.Tensor(i) for i in xi]).to(device)
                outputs = model(xi_tensor)
                proba = torch.cat((proba,softmax(outputs.data)))
                
        return proba.cpu().numpy()


    def predict(self,x,bs=LARGEST_BATCH_SIZE,model=None,device=None):
        if device==None:
            device = self.device
        
        num_batch = x.shape[0]//bs +1*(x.shape[0]%bs!=0)
        pred = torch.zeros(0,dtype=torch.int64).to(device)
         
        if model!=None:
            model = model
        else:
            model = self.load_model()
        model = model.to(device)
        
        torch.set_grad_enabled(False)
        model.eval()        
        
        with torch.no_grad():
            for i in range(num_batch):
                xi = x[i*bs:(i+1)*bs]
                xi_tensor = torch.stack([torch.Tensor(i) for i in xi]).to(device)
                outputs = model(xi_tensor)
                _, predi = torch.max(outputs.data,1)
                pred = torch.cat((pred,predi))

        return pred.cpu().numpy()


    def inference_n_time(self,x,bs=LARGEST_BATCH_SIZE,model=None,device=None):
        if device==None:
            device = self.device
        
        num_batch = x.shape[0]//bs +1*(x.shape[0]%bs!=0)
        pred = torch.zeros(0,dtype=torch.int64).to(device)
         
        if model!=None:
            model = model
        else:
            model = self.load_model()
        model = model.to(device)
        
        torch.set_grad_enabled(False)
        model.eval()        

        import time
        tick = time.time()

        with torch.no_grad():
            for i in range(num_batch):
                xi = x[i*bs:(i+1)*bs]
                xi_tensor = torch.stack([torch.Tensor(i) for i in xi]).to(device)
                outputs = model(xi_tensor)
                _, predi = torch.max(outputs.data,1)
                pred = torch.cat((pred,predi))
        
        return time.time()-tick



    def save(self,checkpoint):
        path = join(self.runs_dir,'checkpoint.pth')
        torch.save(checkpoint,path)

    
    def load_checkpoint(self,filepath):
        if os.path.isfile(filepath):
            checkpoint = torch.load(filepath,map_location=self.device )
            print("Loading ckpt w/ {} epochs".format(checkpoint['epoch']))
            return checkpoint
        else:
            print("No checkpoint foound for ",filepath)
            return None
            
    def load_model(self,inference_mode=True):
        filepath = join(self.runs_dir,'checkpoint.pth')
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

