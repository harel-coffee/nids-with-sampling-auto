import os
from os.path import join 
import numpy as np
import pickle

from sklearn import metrics

import argparse
import pandas as pd
import glob
import time
import ntpath


from utils import ensure_dir,ensure_dirs
from utils import normalize_df, balance_data
from utils import print_processing_time
from utils import print_feature_importance
from utils import shuffle
from utils import get_dtype, get_cols4ml, read_ddos_data
from utils import get_ddos19_mappers
from classifier_settings import get_args, get_classifier, get_balancing_technique
SEED = 234


def get_class_weights(y):
    unique, counts = np.unique(y,return_counts=True)
    weight = 1./counts # make it probabilyt where frequent class has smaller weight
    s = sum(weight)
    weight = weight/s  
    np.set_printoptions(precision=3)
    return weight


def group_data(df): # It will prepare the data into classifer usable format
    grouped = df.groupby(['Flow ID','Label'])
    ID = [ [flowid,label]  for (flowid,label)  in grouped.groups.keys()]
    Label = [label for flowid,label in ID]
    ID = np.array(ID)
    return ID,Label


def df_to_array(df):
    y = df['Label'].values
    #X_df = df.drop(columns=['Hash','Flow ID','Src IP','Src Port','Dst IP','Timestamp', 'Label'])
    X_df = df.drop(columns=['Flow ID', 'Label'])
    X = X_df.values
    return X,y


def encode_label(Y_str,labels_d): 
    Y = [labels_d[y_str] for y_str  in Y_str]
    Y = np.array(Y)
    return np.array(Y)


def train_and_save_classifier(X_train,y_train,args):
    classifier_name = args['classifier_name']
    balance = args['balance']
    clf = get_classifier(args) # runs_dir is a dir to put training log of the classifier   
    
    if balance=='explicit':
        tick = time.time()
        X_train,y_train = balance_data(X_train,y_train)
        tock = time.time()

    tick = time.time()
    print("Shufling data")
    X_train, y_train = shuffle(X_train,y_train)
    print('fitting model')
    clf.fit(X_train,y_train)
    
    if classifier_name in ['tree', 'forest']:
        with open(join(args['runs_dir'],'model.pkl'),'wb') as f:
            pickle.dump(clf,f)

    tock = time.time()
    duration = tock-tick
    print("Trained data of size {} in {:.0f} min, {:.0f} sec ".format(X_train.shape,duration//60,duration%60))
    return


def train(dataroot,classifier_name):
        print("Reading the data...")
        df = read_ddos_data(dataroot) # takes 10GB RAM, loads in 68 seconds
        print("read data of shape ", df.shape)
        label_to_id, id_to_label = get_ddos19_mappers()
        
        balancing_technique = get_balancing_technique()
        input_dim = df.shape[1]-2 # because we remove Label and FlowID columns from X        
        num_class = len(label_to_id.keys())
        WS_flow_count = 13684951 # 13.7 mln records on PCAP-01-12
        num_iters = WS_flow_count*10
        class_weight = None
        classifier_args,config = get_args(classifier_name,WS_flow_count,num_class,input_dim, class_weight,balancing_technique)
        pre_fingerprint = join(dataroot, 'c_{}'.format(classifier_name))
        fingerprint = pre_fingerprint + config
        logdir = join(fingerprint,'log')
        runs_dir = join(logdir,'runs')
        ensure_dir(runs_dir)
        
        df = normalize_df(df,join(runs_dir,'data_stats.pickle'),train_data=True)
        
        X_train, y_train = df_to_array(df)
        y_train = encode_label(y_train,label_to_id)        
        classifier_args['runs_dir'] = runs_dir
        train_and_save_classifier(X_train,y_train,classifier_args)
           
 
if __name__=='__main__':
    datadir = '/data/juma/data/ddos/CSVs_r_1.0/SR_50/RPS_SI_2/PCAP-01-12_l'
    train(datadir,'cnn')
