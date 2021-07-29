import os
from os.path import join as os_join
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
import numpy as np

from sklearn import metrics
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.utils.multiclass import unique_labels
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier

import operator
import argparse
import csv
import pandas as pd
import glob
import time
import ntpath

from models import Classifier

from utils import get_filenames
from utils import make_value2index,ensure_dir,read_data
from utils import print_evaluation,print_absolute_recall, plot_confusion_matrix
from utils import normalize_data, balance_data, get_cm
from utils import print_processing_time
import pickle

SEED = 234
fingerprint = None
use_class_weight_to_balance = True

def get_labels(Y_str):
    labels = sorted(np.unique(Y_str))
    labels_d = make_value2index(labels)
    return labels,labels_d


def correct_data(df, K): # It will prepare the data into classifer usable format
    # please look at the code
    
    # lets get rid of the duplicates
    
    # 1. remove classes less than K items
    print("REMOVING VERY SMALL CLASSES")
    ds = df.Label.value_counts()
    for index,value in ds.items():
        if value<K:
            df = df[df['Label']!=index]
    
    #print("replacing NAN values")
    #df.fillna(df.mean(), inplace=True)

    print("extractining label and flow id")
    df_label = df.loc[:,'Label']
    df_id = df.loc[:,'Flow ID']
    column_names = ["Flow ID","Label"]

    print("now removign them from main df")
    df_data = df.drop(columns=column_names) # now we have 52 features 
    #print(df_data.max())

    #convert categorical to numeric
    print("Convering categorical to numeric")
    categorical_col_names = ["Dst Port","Protocol"]
    for col_name in categorical_col_names:
        ds = df_data[col_name].astype('category')
        df_data[col_name] = ds.cat.codes

    # convert everything to float
    #df_data = df_data.astype(float).apply(pd.to_numeric)
    return df_data.values,df_id.values,df_label.values


def encode_label(Y_str,labels_d): 
    Y = [labels_d[y_str] for y_str  in Y_str]
    Y = np.array(Y)
    return np.array(Y)


def get_classifier(classifier_args,args,runs_dir=None):
    classifier_name = args[0]
    
    if classifier_name=='forest':
        if use_class_weight_to_balance:
            return RandomForestClassifier(n_estimators=10,class_weight='balanced')
        else:
            return RandomForestClassifier(n_estimators=10)
    
   
def process_fold(classifier_args,X_train,y_train,fold_index):
    classifier_name = args[0]
    input_dim = X_train.shape[1]
    num_class = np.unique(y_train)
    clf = get_classifier(classifier_args,args,runs_dir) # runs_dir is a dir to put training log of the classifier    
    #unique,counts = np.unique(y_train,return_counts=True)
    if not use_class_weight_to_balance:
        X_train,y_train = balance_data(X_train,y_train)

    #unique,counts = np.unique(y_train,return_counts=True)
    tick = time.time()
    
    clf.fit(X_train,y_train)
    pickle.dump(clf,open(runs_dir,'wb'))
    tock = time.time()
    duration = tock-tick
    print("Trained and predicted data of sizes ",X_train.shape, X_test.shape," in {0:.2f} min.".format(duration/60))
    print('Done with fold-{}'.format(fold_index))
    importance = [importance for feature_name, importance in zip(feat_names, clf.feat_importances)]

    return importance

def class_report(y,ID):
    unique = np.unique(y)
    for cls_id in unique:
        y_i = y[y==cls_id]
        print(cls_id,y_i.shape)

def process(dataroot,classifier_name,file_ending):
        global K
        K=5        
        data = read_data(dataroot,file_ending)
        print("data is read ",data.shape)
        #data = read_toy_data(dataroot,file_ending)
        X,ID,Y = correct_data(data,K)
        print("data is corrected")    
        labels,labels_d = get_labels(Y)

        X = normalize_data(X)
        print("data is normalized")
        Y = encode_label(Y,labels_d)
        
        input_dim = X.shape[1]
        num_class = len(np.unique(Y))

        confusion_matrix_sum = np.zeros((num_class, num_class),dtype=int)
        if classifier_name=='cnn':
            lr =1e-3
            reg = 1e-5
        elif classifier_name=='softmax':
            lr = 1e-1
            reg = 1e-6
        else:
            lr = None
            reg = None
    
        batch_size =5120 
        device = 'cuda:0'
        #batch_size = 1024*5
        #device = 'cuda:0'
        if use_class_weight_to_balance:
            pre_fingerprint = os_join(dataroot, '{}_k_{}_w'.format(classifier_name,str(K)))
        else:
            pre_fingerprint = os_join(dataroot, '{}_k_{}'.format(classifier_name,str(K)))
            
        optim = 'Adam'
        num_iters = int(Y.shape[0]*10*.8//batch_size) # 10 epochs of whole data. train data is 80% of whole data
        classifier_args = (classifier_name,optim,lr,reg,batch_size,input_dim,num_class,num_iters,device)
        config =  '_optim_{}_lr_{}_reg_{}_bs_{}'.format(optim,lr,reg,batch_size)
        fingerprint = pre_fingerprint + config
        logdir = os_join(fingerprint,'log')
        ensure_dir(fingerprint)       
        ensure_dir(logdir)

        kfold_pred_time = 0
        skf = StratifiedKFold(n_splits=K,random_state=SEED)
        for fold_index, (train_index,test_index) in enumerate(skf.split(X,Y)):
            X_train = X[train_index]
            y_train = Y[train_index]
            X_test = X[test_index]
            test_id = ID[test_index]
            y_test = Y[test_index]
            runs_dir=os_join(logdir,'fold_{}'.format(fold_index))
            pred,duration = classify_fold(classifier_name,X_train,y_train,X_test,fold_index,classifier_args,runs_dir)
            acc = metrics.balanced_accuracy_score(y_test,pred)
            print("Balanced accuracy including benign: {}".format(acc))
            kfold_pred_time+=duration
            assert pred.shape==y_test.shape, "y_true={} and pred.shape={} should be same ".format(y_test.shape,pred.shape)
            plot_confusion_matrix(os_join(fingerprint,'cm_fold_{}.jpg'.format(fold_index)), y_test, pred, classes=labels,normalize=False, title='Confusion matrix, with normalization')
            plot_confusion_matrix(os_join(fingerprint,'cm_norm_fold_{}.jpg'.format(fold_index)), y_test, pred, classes=labels,normalize=True, title='Confusion matrix, with normalization')
            cm_i = confusion_matrix(y_test,pred)
            confusion_matrix_sum+=cm_i
        cm = np.array(confusion_matrix_sum/K).astype(np.float)
        print(dataroot,classifier_name)
        plot_confusion_matrix(os_join(fingerprint,'cm_nonnorm_fold_avg.jpg'), [], [],cm=cm, classes=labels, title='Confusion matrix, without normalization')
        plot_confusion_matrix(os_join(fingerprint,'cm_norm_fold_avg.jpg'), [], [],cm=cm, classes=labels,normalize=True, title='Confusion matrix, with normalization')
 
        print_evaluation(cm, labels, os_join(fingerprint,'evaluation.csv'))
        print_absolute_recall(cm, labels, os_join(fingerprint,'absolute_recall.csv'))


if __name__=='__main__':
    tick = time.time()    
    classify('/home/juma/data/net_intrusion/CIC-IDS-2018/CSVs/without_sampling_l', 'cnn','*Meter.csv') 
    tock = time.time()
    print("Time taken {}".format(tock-tick))
