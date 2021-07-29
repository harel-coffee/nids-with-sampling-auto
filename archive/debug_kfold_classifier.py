import os
from os.path import join 
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pickle

from sklearn import metrics
from sklearn.metrics import classification_report,confusion_matrix
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

import argparse
import pandas as pd
import glob
import time
import ntpath

from models import Classifier

from utils import get_filenames
from utils import make_value2index, ensure_dir, ensure_dirs, read_data
from utils import print_evaluation, print_absolute_recall, plot_confusion_matrix
from utils import normalize_data, balance_data, get_cm
from utils import print_processing_time
from utils import print_feature_importance


SEED = 234
fingerprint = None

def get_labels(Y_str):
    labels = sorted(np.unique(Y_str))
    labels_d = make_value2index(labels)
    return labels,labels_d


def get_class_weights(y):
    unique, counts = np.unique(y,return_counts=True)
    print(np.asarray((unique,counts)).T)
    #class_samples_count = np.array([len(np.where(y==t)) for t in np.unique(y)])
    #print("class_samples_count = \n",counts)
    weight = 1./counts # make it probabilyt that sums to 1
    s = sum(weight)
    weight = weight/s  
    np.set_printoptions(precision=5)
    print("normalized weights ", weight)
    return weight


def group_data(df): # It will prepare the data into classifer usable format
   
    print("Grouping by FlowID and Label")
    grouped = df.groupby(['Flow ID','Label'])
    ID = [ [flowid,label]  for (flowid,label)  in grouped.groups.keys()]
    Label = [label for flowid,label in ID]
    ID = np.array(ID)
    return ID,Label


def df_to_array(df):
    y = df['Label'].values
    X_df = df.drop(columns=['Flow ID', 'Label'])
    X = X_df.values
    return X,y


def encode_label(Y_str,labels_d): 
    Y = [labels_d[y_str] for y_str  in Y_str]
    Y = np.array(Y)
    return np.array(Y)


def get_classifier(args,runs_dir=None):
    classifier_name = args['classifier_name'] 
    balance = args['balance']

    if classifier_name=='tree':
        if balance=='with_loss':
            return tree.DecisionTreeClassifier(class_weight='balanced')
        elif balance=='explicit':
            return tree.DecisionTreeClassifier()
        elif balance =='sample_per_batch':
            print("Decision Tree does not use mini batch")
            exit()
    
    elif classifier_name=='forest':
        if balance=='with_loss':
            return RandomForestClassifier(n_estimators=10,class_weight='balanced')
        elif balance=='explicit':
            return RandomForestClassifier(n_estimators=10)
        elif balance=='sample_per_batch':
            print("Forst does not use mini batch")
            exit()
    
    elif classifier_name=='softmax':
        if runs_dir is None:
            print('please provide runs_dir')
            exit(1)
        clf = Classifier(method='softmax',input_dim = args['input_dim'],num_classes=args['num_class'],lr=args['lr'],reg=args['reg'],num_iters=args['num_iters'],batch_size=args['batch_size'],runs_dir=runs_dir,device=args['device'],balance=args['balance'], class_weights=args['class_weight'])
        return clf

    elif classifier_name=='cnn':
        if runs_dir is None:
            print('please provide runs_dir')
            exit(1)
        clf = Classifier(method='cnn2',input_dim = args['input_dim'],num_classes=args['num_class'],lr=args['lr'],reg=args['reg'],num_iters=args['num_iters'],batch_size=args['batch_size'],runs_dir=runs_dir,device=args['device'],balance=args['balance'],class_weights=args['class_weight'])
        return clf


def train_fold(X_train,y_train,fold_index,args,runs_dir):
    classifier_name = args['classifier_name']
    balance = args['balance']
    clf = get_classifier(args,runs_dir) # runs_dir is a dir to put training log of the classifier    
    if balance=='explicit':
        print("explicit balancing data")
        X_train,y_train = balance_data(X_train,y_train)

    tick = time.time()
    #clf.fit(X_train,y_train)
    
    if classifier_name in ['tree', 'forest']:
        pickle.dump(clf,open(runs_dir,'wb'))

    tock = time.time()
    duration = tock-tick
    print("Trained data of size {} for Fold-{} in {:.0f} min, {:.0f} sec ".format(X_train.shape,fold_index,duration//60,duration%60))
    return clf,duration


def predict_fold(classifier_name,clf,df,ids, y_test):
    grouped = df.groupby(['Flow ID','Label'])
    flow_pred_any = []
    flow_pred_majority = []
    pred_i = None
    for (flowid,label),y in zip(ids,y_test):
        frames = grouped.get_group((flowid,label))
        frames = frames.drop(columns=['Flow ID','Label'])
        if classifier_name in ['cnn','softmax']:
            pred_i = clf.predict(frames.values,eval_mode = True)
        else:
            pred_i = clf.predict(frames.values)

        if 'any'=='any':
            if (pred_i==y).sum()>0:
                flow_pred_any.append(y) # if any of the records are detected, we consider flow is detected
            else:
                counts = np.bincount(pred_i)
                flow_pred_any.append(np.argmax(counts)) # else, most common misprediction label is given to flow
        if 'majority'=='majority':
            counts = np.bincount(pred_i)
            flow_pred_majority.append(np.argmax(counts)) # most common prediction label is given to flow
    return np.array(flow_pred_any),np.array(flow_pred_majority)
   

def class_report(y,ID):
    unique = np.unique(y)
    for cls_id in unique:
        y_i = y[y==cls_id]
        print(cls_id,y_i.shape)


def classify(dataroot,classifier_name):
        global K
        K=5
        #fraction = 1
        fraction = 0.001
        
        #total_records_in_whole = 6907723;
        total_records = 6907705; # in fold
        folds_df = []
        fold_root = join(dataroot,'folds_fraction_{}'.format(fraction))
        
        ds_list = []
        for fold_index in range(K):
            df = pd.read_csv(join(fold_root,'fold_{}.csv'.format(fold_index))) 
            folds_df.append(df)
            ds_list.append(df.Label)
        total_label_df = pd.concat(ds_list,sort=False)
        labels,labels_d = get_labels(total_label_df.unique())
        class_weight = get_class_weights(encode_label(total_label_df.values,labels_d))
        
        #balance = 'sample_per_batch'
        #balance = 'with_loss'
        balance = 'explicit'
 
        input_dim = folds_df[0].shape[1]-2 # because we remove Label and FlowID columns from X
        labels,labels_d = get_labels(folds_df[0].Label.unique())

        num_class = len(labels)

        if classifier_name in ['cnn','softmax']:
            batch_size =256
            num_iters = 0.1*(total_records*.8*.9)//batch_size # 10 epochs for total dataset
            optim='Adam'  
            if classifier_name=='cnn':
                lr =1e-3
                reg = 0
                device = [0,1]

            elif classifier_name=='softmax':
                lr = 1e-3
                reg =0 
                device = 'cuda:0'
            classifier_args = {'classifier_name':classifier_name,'optim':optim,'lr':lr,'reg':reg,'batch_size':batch_size,'input_dim':input_dim,'num_class':num_class,'num_iters':num_iters,'device':device, 'balance':balance, 'class_weight':class_weight}
            config =  '_optim_{}_lr_{}_reg_{}_bs_{}_b_{}'.format(optim,lr,reg,batch_size,balance)
        else:
            lr = None
            reg = None
            batch_size=None
            device=None
            classifier_args = {'classifier_name':classifier_name,'balance':balance}
            config = '_b_{}'.format(balance)
        
        
        pre_fingerprint = join(dataroot,'classifiers','kfold', 'r_{}_c_{}_k_{}'.format(fraction,classifier_name,str(K)))
            
        fingerprint = pre_fingerprint + config
        print("Running experiment \n ",fingerprint)
        logdir = join(fingerprint,'log')
        cmdir = join(fingerprint,'cm')
        recalldir = join(fingerprint,'recall')
        evaldir = join(fingerprint,'eval')

        ensure_dirs(fingerprint,logdir,cmdir,recalldir,evaldir)       

        confusion_matrix_sum = np.zeros((num_class, num_class),dtype=float)
        majority_confusion_matrix_sum = np.zeros((num_class, num_class),dtype=float)
        kfold_pred_time = 0
        kfold_feature_importance = np.zeros(input_dim,dtype=np.float)
        skf = StratifiedKFold(n_splits=K,random_state=SEED)
        for fold_index in range(K):
            print("Fold ",fold_index)
            test_df = folds_df[fold_index]
            train_df = pd.concat([folds_df[i] for i in range(K) if i!=fold_index],sort=False)
            
            X_train, y_train = df_to_array(train_df)
            y_train = encode_label(y_train,labels_d)

            runs_dir=join(logdir,'fold_{}'.format(fold_index))
            clf,duration = train_fold(X_train,y_train,fold_index,classifier_args,runs_dir)
            if classifier_name=='forest':
                kfold_feature_importance+=clf.feature_importances_
            
            flowids_test,y_flowid_test = group_data(test_df)
            y_flowid_test = encode_label(y_flowid_test,labels_d)
            pred_any, pred_majority = predict_fold(classifier_name,clf,test_df, flowids_test, y_flowid_test)
            
            assert pred_any.shape==pred_majority.shape,"any and majority shapes should be same {},{}".format(pred_any.shape,pred_majority.shape)
            #assert pred.shape==y_flowid_test.shape, "y_true={} and pred.shape={} should be same ".format(y_flowid_test.shape,pred.shape)
            acc_pred_any = metrics.balanced_accuracy_score(y_flowid_test,pred_any)
            acc_pred_majority = metrics.balanced_accuracy_score(y_flowid_test,pred_majority)
            print("Fold Balanced accuracy(any,majority): ({:.2f},{:.2f})".format(acc_pred_any,acc_pred_majority))
            kfold_pred_time+=duration

            plot_confusion_matrix(join(cmdir,'any_fold_{}.jpg'.format(fold_index)), y_flowid_test, pred_any, classes=labels,normalize=False, title='Confusion matrix, with normalization')
            plot_confusion_matrix(join(cmdir,'any_norm_fold_{}.jpg'.format(fold_index)), y_flowid_test, pred_any, classes=labels,normalize=True, title='Confusion matrix, with normalization')         
            cm_i = confusion_matrix(y_flowid_test,pred_any)
            print_absolute_recall(cm_i, labels, join(recalldir,'any_fold_{}.csv'.format(fold_index)),fold_root)
            print_evaluation(cm_i, labels, join(evaldir,'any_fold_{}.csv'.format(fold_index)))
            confusion_matrix_sum+=cm_i
       

            plot_confusion_matrix(join(cmdir,'majority_fold_{}.jpg'.format(fold_index)), y_flowid_test, pred_majority, classes=labels,normalize=False, title='Confusion matrix, with normalization')
            plot_confusion_matrix(join(cmdir,'majority_norm_fold_{}.jpg'.format(fold_index)), y_flowid_test, pred_majority, classes=labels,normalize=True, title='Confusion matrix, with normalization')                       
            majority_cm_i = confusion_matrix(y_flowid_test,pred_majority)
            print_absolute_recall(majority_cm_i, labels, join(recalldir,'majority_fold_{}.csv'.format(fold_index)),fold_root)
            print_evaluation(majority_cm_i, labels, join(evaldir,'majority_fold_{}.csv'.format(fold_index)))
            majority_confusion_matrix_sum+=majority_cm_i

        if classifier_name=='forest':
            print_feature_importance(kfold_feature_importance,join(dataroot,'folds_fraction_{}'.format(fraction),'feature_selection.csv'))


        cm = confusion_matrix_sum
        cm_majority = majority_confusion_matrix_sum
        print(dataroot,classifier_name)

 
        plot_confusion_matrix(join(cmdir,'avg_any_fold.jpg'), [], [],cm=cm, classes=labels, title='Confusion matrix, without normalization')
        plot_confusion_matrix(join(cmdir,'avg_any_norm_fold.jpg'), [], [],cm=cm, classes=labels, normalize=True, title='Confusion matrix, with normalization')

        
        plot_confusion_matrix(join(cmdir,'avg_majority_fold.jpg'), [], [],cm=cm_majority, classes=labels, title='Confusion matrix, without normalization')
        plot_confusion_matrix(join(cmdir,'avg_majority_norm_fold.jpg'), [], [],cm=cm_majority, classes=labels, normalize=True, title='Confusion matrix, with normalization')
        
        print_evaluation(cm, labels, join(fingerprint,'evaluation_any.csv'))
        print_evaluation(cm_majority, labels, join(fingerprint,'evaluation_majority.csv'))
        
        print_absolute_recall(cm, labels, join(fingerprint,'recall_any.csv'),fold_root,kfold=True)
        print_absolute_recall(cm_majority, labels, join(fingerprint,'recall_majority.csv'),fold_root,kfold=True)

if __name__=='__main__':
    tick = time.time()    
    classify('/home/juma/data/net_intrusion/CIC-IDS-2018/CSVs/without_sampling_l', 'cnn') 
    tock = time.time()
    print("Time taken {:.0f} min. {} sec.".format((tock-tick)//60,(tock-tick)%60))
