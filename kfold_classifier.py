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
from utils import get_ids18_mappers, get_class_weights
from classifier_settings import get_args, get_classifier, get_balancing_technique
from utils import get_cols4ml, encode_label, get_dtype, get_dtype4normalized


def load_folds(dataroot, fold_prefix, K):
    df_list =  [pd.read_csv(join(dataroot,fold_prefix.format(i)), usecols=get_cols4ml(), dtype=get_dtype4normalized()) for i in range(K)]
  
    fold_data = [ (df.drop(columns=['Label']).values, encode_label(df.Label.values)) \
    for df in df_list]
    return fold_data


def train(dataroot,classifier_name='cnn'):
        balance = get_balancing_technique()
        K = 10
        fold_prefix = str(K)+'bal_fold_{}.csv' if balance=='explicit' else str(K) + 'r_fold_{}.csv'

        class_weight = get_class_weights(dataroot)

        classifier_args, config = get_args(classifier_name, class_weight )
        pre_fingerprint = join(dataroot, 'c_{}'.format(classifier_name))
        fingerprint = join(pre_fingerprint + config, 'K_{}'.format(K))
        print(fingerprint)
        folds_data = load_folds(dataroot, fold_prefix, K) 
        for test_index in range(K):
            print('-----------{}----------'.format(test_index))
            X_train = np.concatenate([fold[0] for i, fold in enumerate(folds_data) if i!=test_index ],axis=0)
            y_train = np.concatenate([fold[1] for i, fold in enumerate(folds_data) if i!=test_index ],axis=0)
            
            logdir = join(fingerprint,'log','{}'.format(test_index))
            ensure_dir(logdir)
            classifier_args['runs_dir'] = logdir
            clf = get_classifier(classifier_args)
            clf.fit(X_train, y_train) 
            modelname = join(classifier_args['runs_dir'],'model.pkl')
            pickle.dump(clf, open(modelname,'wb'))

if __name__=='__main__':
    if 'noWS' == 'WS':
        #datadir = '/data/juma/data/ids18/CSVs/WS_l'
        datadir = '/hdd/juma/data/net_intrusion/ids18/CSVs_r_1.0_m_1.0/WS_l'
        classify(datadir, 'tree')
    else:
        csv_dir = 'CSVs_r_1.0_m_1.0'
        sampler_dirs = {}
        sampler_dirs['SI_10'] = ['SFS_SI_9.77_l', 'SGS_e_0.05_l','SRS_SI_10_l','FFS_(8,16,4)_l']
        sampler_dirs['SI_100'] = ['SFS_SI_95.33_l', 'SGS_e_1_l','SRS_SI_100_l','FFS_(8,16,40)_l']
        sampler_dirs['SI_1000'] = ['SFS_SI_685.08_l','SRS_SI_1000_l','SGS_e_11.5_l','FFS_(8,16,400)_l']
        samp_inter = 'SI_1000'
        for d in sampler_dirs[samp_inter]:
                datadir = '/data/juma/data/ids18/{}/{}/{}'.format(csv_dir,samp_inter,d)
                print(d)
                train(datadir,'tree')
