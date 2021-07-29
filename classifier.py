import os
from os.path import join 
import numpy as np
import pickle

from sklearn import metrics
from sklearn.metrics import confusion_matrix

import argparse
import pandas as pd
import glob
import time
import ntpath
from tqdm import tqdm

from utils import ensure_dir
from utils import get_ids18_mappers
from classifier_settings import get_args, get_classifier, get_balancing_technique
from utils import get_cols4ml, encode_label, get_dtype, get_dtype4normalized


def get_class_weights(y,p=.2): # only useful when cost-based
    unique, counts = np.unique(y,return_counts=True)
    # adjust for reduced benign
    counts[0] = int(counts[0]*p)
    weight = 1./counts # make it probabilyt where frequent class has smaller weight
    s = sum(weight)
    weight = weight/s  
    np.set_printoptions(precision=3)
    return weight


def train_fold(X_train,y_train,args):
    classifier_name = args['classifier_name']
    runs_dir = args['runs_dir']

    clf = get_classifier(args) # runs_dir is a dir to put training log of the classifier   

    print('fitting the model')
    tick = time.time()
    clf.fit(X_train,y_train)
    tock = time.time()
    duration = tock-tick
    print("Trained data of size {} in {:.0f} min, {:.0f} sec ".format(X_train.shape,duration//60,duration%60))

    if classifier_name in ['tree', 'forest']:
        fn = runs_dir+'.pkl'
        print("Saving to ",fn)
        pickle.dump(clf,open(fn,'wb'))

    return clf,duration


def classify(dataroot,classifier_name):
        K=5
        balance = get_balancing_technique()
        train_data = []
        #single fold 29M records
        # 4 folds 120M records
        # if 20M records require 5% RAM
        # then 120M records require 30% memory
        print("Reading the data...")
        tick=time.time()
        label_to_id, id_to_label, _ = get_ids18_mappers()
        num_train_records = 0
        print("Reading 4 folds ")
        
        if balance=='with_loss' or balance=='no' or balance=='with_loss_sub': 
            regex  = 'r_fold_{}.csv'
        elif balance=='explicit':
            regex = 'bal_fold_{}.csv'
            
        for fold_index in tqdm(range(K)):
            if fold_index==0:
                continue
            reader = pd.read_csv(join(dataroot,regex.format(fold_index)),chunksize=10**6, usecols=get_cols4ml(), dtype=get_dtype4normalized())# 10**6 rows read in 9min, total is 29*10**6
            # remove the extra header row
            for df in tqdm(reader):
                y_str = df.Label.values
                x = df.drop(columns=['Label']).values
                train_data.append((x,encode_label(y_str)))
                num_train_records +=df.shape[0]
                print(df.memory_usage(deep=True).sum()*(799902)/(1024*1024*1024 ))
        tock = time.time()
        print("read data in {:.2f}".format(tock-tick)) # 24min

        classifier_args, config = get_args(classifier_name, num_class='dummy', class_weight=None)
        pre_fingerprint = join(dataroot, 'c_{}'.format(classifier_name))
            
        fingerprint = pre_fingerprint + config
        logdir = join(fingerprint,'log')
        ensure_dir(logdir)
               
        X_train = np.concatenate([fold[0] for fold in train_data ],axis=0)
        y_train = np.concatenate([fold[1] for fold in train_data ],axis=0)
        classifier_args['runs_dir']=logdir

        print("Start training")
        tick = time.time()
        clf= get_classifier(classifier_args)
        print("classes")
        print(np.unique(y_train))
        clf.fit(X_train, y_train)
        fn = classifier_args['runs_dir']+'.pkl'
        pickle.dump(clf,open(fn,'wb'))
        print("Done training {} flow records in {:.2f} sec".format(y_train.shape[0],time.time()-tick))

if __name__=='__main__':
    if 'WS' == 'WS':
        #datadir = '/data/juma/data/ids18/CSVs/WS_l'
        datadir = '/hdd/juma/data/net_intrusion/ids18/CSVs_r_1.0_m_1.0/WS_l'
        classify(datadir, 'forest')
    else:
        csv_dir = 'CSVs_r_1.0_m_1.0'

        #sampler_dirs = ['SFS_SI_9.77_l','FFS_(8,16,4)_l','SGS_e_0.05_l','SRS_SI_10_l']
        #sampler_dirs = ['SFS_SI_95.33_l', 'SGS_e_1_l','SRS_SI_100_l','FFS_(8,16,40)_l']
        sampler_dirs = ['SFS_SI_685.08_l','SRS_SI_1000_l','SGS_e_11.5_l','FFS_(8,16,400)_l']
        sdir = 'SR_0.1'
        for d in sampler_dirs:
            datadir = '/data/juma/data/ids18/{}/{}/{}'.format(csv_dir,sdir,d)
            classify(datadir,'forest')
