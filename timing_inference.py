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
from tqdm import tqdm

from utils import  ensure_dir, read_data
from utils import print_processing_time
from utils import print_feature_importance
from utils import shuffle
from utils import get_ids18_mappers
from classifier_settings import get_args, get_classifier, get_balancing_technique


SEED = 234
def get_class_weights(y):
    unique, counts = np.unique(y,return_counts=True)
    weight = 1./counts # make it probabilyt where frequent class has smaller weight
    s = sum(weight)
    weight = weight/s  
    return weight


def encode_label(Y_str,labels_d): 
    Y = [labels_d[y_str] for y_str  in Y_str]
    Y = np.array(Y)
    return np.array(Y)


def time_inference(classifier_name,clf,df,dataroot):
    df = df.drop(columns=['Flow ID','Label'])
    X = df.values
    tick = time.time()
    if classifier_name in ['cnn','softmax']:
        pred = clf.predict(X,eval_mode=True)
    else:
        pred = clf.predict(X)
    tock = time.time()
    prediction_time = tock - tick
    print("Predicted data of size {}(samples/second) in {:.0f}({:.0f})sec".format(X.shape[0], 1, prediction_time, X.shape[0]/prediction_time))
    with open(join(dataroot,'inference_time.txt'),'w') as f:
        f.write('{}'.format(prediction_time))


def get_runs_dir(logdir):#replaces runs_dir with CSVs_r_1.0 runs_dir
            fold_index=0
            runs_dir=join(logdir,'fold_{}'.format(fold_index))
            # for mem constrained experiemnt II, we need same classifier CSVs_r_1 for all memories
            start = runs_dir.find('CSVs_r_')
            end = runs_dir.find('SR_10')
            CSV_dirname = runs_dir[start:end-1]
            return runs_dir.replace(CSV_dirname,'CSVs_r_1.0')


def classify(dataroot,classifier_name):
        K=5
        fraction = 1
        
        #total_records = 6907705; # in fold fraction after removin small classes <K
        folds_df = []
        fold_root = join(dataroot,'folds_fraction_{}'.format(fraction))
        print("Reading the data...")
        ds_list = [] 
        for fold_index in range(K):
            df = pd.read_csv(join(fold_root,'fold_{}.csv'.format(fold_index))) 
            folds_df.append(df)
            ds_list.append(df.Label)
        total_df = pd.concat(folds_df)  
        total_label_df = pd.concat(ds_list)
        labels = total_label_df.sort_values().unique()
        total_records = total_label_df.shape[0]
        #labels,labels_d = get_labels(total_label_df.unique())
        label_to_id, id_to_label, _  = get_ids18_mappers()
        class_weight = get_class_weights(encode_label(total_label_df.values,label_to_id))
        
        balance = get_balancing_technique()
        input_dim = folds_df[0].shape[1]-2 # because we remove Label and FlowID columns from X
        gt_num_class = len(label_to_id)
        num_class = len(labels)
        assert gt_num_class==num_class, 'all classess should be observed gt_classes!=observed_classes {}!={}'.format(gt_num_class,num_class)

        classifier_args,config = get_args(classifier_name, total_records,gt_num_class, input_dim,class_weight, balance)
        pre_fingerprint = join(dataroot, 'r_{}_c_{}_k_{}'.format(fraction,classifier_name,str(K)))
            
        fingerprint = pre_fingerprint + '_mem_constrained' + config
        logdir = join(pre_fingerprint+config,'log')
        runs_dir = get_runs_dir(logdir)
        classifier_args['runs_dir'] = runs_dir
        clf = get_classifier(classifier_args)
        time_inference(classifier_name,clf,total_df,dataroot)


if __name__=='__main__':
    datadir = '/data/juma/data/ids18/CSVs_r_0.001/SR_10/SEL_(9500,1,1)_l'
    classify(datadir,'cnn')
