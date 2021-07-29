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
import subprocess

from utils import ensure_dir
from utils import get_ids18_mappers
from classifier_settings import get_args, get_classifier, get_balancing_technique

SEED = 234

def get_record_count(dataroot):
    df = pd.read_csv(join(dataroot,'label_dist.csv'), names=['Label','Count'])
    return df['Count'].sum()


def get_class_weights(dataroot,p=1):
    df = pd.read_csv(join(dataroot,'label_dist.csv'), names=['Label','Count'])
    label_to_id, id_to_label, _ = get_ids18_mappers()
    #order of labels should be same as label ids on train data 
    counts = []
    print(id_to_label)
    for i in range(len(id_to_label)):
        label = id_to_label[i]
        if label in df['Label'].values: 
            c = df[df['Label']==label]['Count'].iloc[0]
            counts.append(c)
        else:
            print('not found', label)
            counts.append(0)
 
    counts = np.array(counts)
    normed_weights = [1-(count/sum(counts)) for count in counts]
    return np.array(normed_weights)

    weight = 1./counts # make it probabilyt where frequent class has smaller weight
    s = sum(weight)
    weight = weight/s # normalization  
    np.set_printoptions(precision=3)
    print("class weights = ", weight)
    return weight


def classify(dataroot,classifier_name='cnn'):
        class_weight = get_class_weights(dataroot)
        balance = get_balancing_technique()
        print('balancing technique ', balance)
        if balance=='explicit':
            train_csv = join(dataroot,'bal_train.csv')    
            val_csv = join(dataroot,'bal_fold_1.csv') # no need to use bal__fold because it is shuffled
        else :
            train_csv = join(dataroot,'r_train.csv')
            val_csv = join(dataroot,'r_fold_1.csv')
        
        result_val = subprocess.run(['wc','-l',val_csv], stdout=subprocess.PIPE)
        result_train = subprocess.run(['wc','-l',train_csv], stdout=subprocess.PIPE)
        train_records = int(result_train.stdout.split()[0])-1 # for the header
        val_records = int(result_val.stdout.split()[0])-1
        print("Number of train and val records ({},{})".format(train_records, val_records))
        
        num_epochs = 40
        label_to_id, id_to_label, _ = get_ids18_mappers()
        #class_weight = None
        class_weight = get_class_weights(dataroot)
        if balance=='with_loss_inverse':
            class_weight = 1./class_weight

        num_class = len(label_to_id) # we assume all the categories are observed

        classifier_args, config = get_args(classifier_name, num_class, class_weight )
        pre_fingerprint = join(dataroot, 'c_{}'.format(classifier_name))
        fingerprint = pre_fingerprint + config
        logdir = join(fingerprint,'log')
        ensure_dir(logdir)
        classifier_args['runs_dir'] = logdir
        clf = get_classifier(classifier_args)         
        clf.fit(train_csv, val_csv,num_epochs,train_records, val_records)


if __name__=='__main__':
    case='noWS'
    if case=='WS':
        datadir = '/data/juma/data/ids18/CSVs_r_1.0_m_1.0/WS_l'
        #datadir = '/hdd/juma/data/net_intrusion/ids18/CSVs/WS_l'
        classify(datadir)
    else:
        csv_dir = 'CSVs_r_1.0_m_1.0'
        sampler_dirs = ['SFS_SI_9.77_l', 'SGS_e_0.05_l','SRS_SI_10_l','FFS_(8,16,4)_l']
        #sampler_dirs = ['SFS_SI_95.33_l', 'SGS_e_1_l','SRS_SI_100_l','FFS_(8,16,40)_l']
        #sampler_dirs = ['SFS_SI_930.75_l','SRS_SI_1000_l','SGS_e_10_l','FFS_(8,16,400)_l']  
        sdir = 'SR_10.0'
        for d in sampler_dirs:
            datadir = '/data/juma/data/ids18/{}/{}/{}'.format(csv_dir,sdir,d)
            classify(datadir,'cnn')

