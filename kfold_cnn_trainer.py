import os
from os.path import join 
import numpy as np

from sklearn import metrics
from sklearn.metrics import confusion_matrix

import pandas as pd
import glob
import time
import ntpath
from tqdm import tqdm

from utils import ensure_dir
from utils import get_ids18_mappers, get_class_weights
from classifier_settings import get_args, get_classifier, get_balancing_technique

def train(dataroot,classifier_name='cnn'):
        balance = get_balancing_technique()
        K = 10
        fold_prefix = '{}bal_fold_{}.csv' if balance=='explicit' else '{}r_fold_{}.csv'

        class_weight = get_class_weights(dataroot)
        
        classifier_args, config = get_args(classifier_name, class_weight )
        pre_fingerprint = join(dataroot, 'c_{}'.format(classifier_name))
        fingerprint = join(pre_fingerprint + config, 'K_{}'.format(K))
        print(fingerprint)
        num_epochs = 40
        for test_index in range(K):
            print('-----------{}----------'.format(test_index))
            dev_indices = [i for i in range(K) if i!=test_index]
            val_index = dev_indices[0]
            train_indices = dev_indices[1:]
            val_csv = join(dataroot,fold_prefix.format(K,val_index))
            list_of_train_csvs = [join(dataroot,fold_prefix.format(K,i)) for i in train_indices]

       
            logdir = join(fingerprint,'log','{}'.format(test_index))
            ensure_dir(logdir)
            classifier_args['runs_dir'] = logdir
            clf = get_classifier(classifier_args)         
            clf.fit(list_of_train_csvs, val_csv,num_epochs)


if __name__=='__main__':
    case='noWS'
    if case=='WS':
        datadir = '/data/juma/data/ids18/CSVs_r_1.0_m_1.0/WS_l'
        #datadir = '/hdd/juma/data/net_intrusion/ids18/CSVs/WS_l'
        classify(datadir)
    else:
        csv_dir = 'CSVs_r_1.0_m_1.0'
        sampler_dirs = {}
        sampler_dirs['SI_10'] = ['SFS_SI_9.77_l', 'SGS_e_0.05_l','SRS_SI_10_l','FFS_(8,16,4)_l']
        sampler_dirs['SI_100'] = ['SFS_SI_95.33_l', 'SGS_e_1_l','SRS_SI_100_l','FFS_(8,16,40)_l']
        sampler_dirs['SI_1000'] = ['SFS_SI_685.08_l','SRS_SI_1000_l','SGS_e_11.5_l','FFS_(8,16,400)_l']
        samp_inter = 'SI_1000'
        for d in sampler_dirs[samp_inter][:1]:
                datadir = '/data/juma/data/ids18/{}/{}/{}'.format(csv_dir,samp_inter,d)
                print(d)
                train(datadir,'cnn')



