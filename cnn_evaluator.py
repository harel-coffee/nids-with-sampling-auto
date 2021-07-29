import os
from os.path import join 
import numpy as np

from sklearn import metrics
from sklearn.metrics import confusion_matrix

import argparse
import pandas as pd
import glob
import time
import ntpath
from tqdm import tqdm
import subprocess

from utils import get_cols4eval, get_dtype4normalized
from utils import get_ids18_mappers
from utils import encode_label
from classifier_settings import get_args, get_classifier
from classifier_settings import ClassifierLoader
from bookkeepers import result_logger_ids18


def group_data(df): # It will prepare the data into classifer usable format
    grouped = df[['Flow ID', 'Day','Label','Protocol']].groupby(['Flow ID','Day','Label'], sort=True)
    group_ids = [ [flowid,day, label]  for (flowid,day, label)  in sorted(grouped.groups.keys())]
    group_labels = [label for flowid, day, label in group_ids]
    return np.array(group_ids), group_labels, grouped


def get_class_weights(dataroot,p=1):
    df = pd.read_csv(join(dataroot,'label_dist.csv'), names=['Label','Count'])
    label_to_id, id_to_label, _ = get_ids18_mappers()
    #order of labels should be same as label ids on train data
    counts = []
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



def evaluate_per_flow(clf,y, grouped, ordered_df):
    record_count_ls = grouped['Label'].agg('count')
    X = ordered_df.drop(columns=['Flow ID','Label','Day','Timestamp']).values
    
    # prediction
    print("Start prediction")
    tick = time.time()
    pred = clf.predict(X)
    tock = time.time()
    prediction_time = tock - tick
    print("Predicted in {:.2f}".format(prediction_time))

    # evaluate prediction per-flow (5 tuple)
    flow_pred_any = []
    flow_pred_majority = []
    flow_pred_all = []
    p_records = 0 # processed_records
    tick = time.time()
    for i,n_records in tqdm(enumerate(record_count_ls)):
        pred_i = pred[p_records:p_records+n_records]
        counts = np.bincount(pred_i)
        most_common_pred = np.argmax(counts)
        if 'any'=='any':
            if (pred_i==y[i]).sum()>0:
                flow_pred_any.append(y[i]) # if any of the records are detected, we consider flow is detected
            else:
                flow_pred_any.append(most_common_pred) # else, most common misprediction label is given to flow
        if 'majority'=='majority':
            flow_pred_majority.append(most_common_pred) # most common prediction label is given to flow
        if 'all'=='all':
            if np.all(y[i]==pred_i):
                flow_pred_all.append(y[i])
            else:
                error_counts = np.bincount(pred_i[np.where(pred_i!=y[i])])
                most_common_error = np.argmax(error_counts)
                flow_pred_all.append(most_common_error)
        p_records+=n_records
    tock = time.time()
    eval_duration = tock-tick
    print("Time for Evaluation: {:.0f} min {} sec".format((eval_duration)//60,(eval_duration)%60)) # 20 min for SRS
    return np.array(flow_pred_any),np.array(flow_pred_majority), np.array(flow_pred_all), eval_duration


def evaluator(dataroot,classifier_name):
        print('evaluating ', ntpath.basename(dataroot))

        test_csv_file = join(dataroot,'fold_0.csv')
        result_test = subprocess.run(['wc','-l', test_csv_file], stdout=subprocess.PIPE)
        test_records = int(result_test.stdout.split()[0])
 
        # load Classifier
        class_weight = get_class_weights(dataroot) # because it is not important for evaluation
        num_class = 14 # because we remove Label,FlowID,Timestamp columns from X
        classifier_args, config = get_args(classifier_name, num_class,class_weight)

        pre_fingerprint = join(dataroot, 'c_{}'.format(classifier_name))
        fingerprint = pre_fingerprint + config
        print('clf fingerprint', ntpath.basename(fingerprint))  
        classifier_args['runs_dir'] = join(fingerprint,'log')
        clf = ClassifierLoader().load(classifier_args)
        # classifier loaded

        # load data
        col_names = get_cols4eval()
        col_names.append('Timestamp')
        df = pd.read_csv(test_csv_file, usecols=col_names, dtype=get_dtype4normalized())
        print("Record distribution:")
        print(df.Label.value_counts())
        df['Day'] = df['Timestamp'].map(lambda x: x[:2]).astype(str) # type string
       
        #group data 
        df = df.sort_values(by=['Flow ID','Label'])# replaces ordering task in per_flow_eval
        flowids,flowlabels, grouped = group_data(df)
        y = encode_label(flowlabels)
        print("data is grouped and labels are encoded")
        pred_any, pred_maj, pred_all, _ = evaluate_per_flow(clf,y,grouped, df)

        any_cm = confusion_matrix(y,pred_any)
        maj_cm = confusion_matrix(y,pred_maj)
        all_cm = confusion_matrix(y,pred_all)

        any_acc = metrics.balanced_accuracy_score(y, pred_any)
        maj_acc = metrics.balanced_accuracy_score(y, pred_maj)
        all_acc = metrics.balanced_accuracy_score(y, pred_all)
        print(any_acc, maj_acc, all_acc)

        result_logger_ids18(fingerprint, np.unique(y),(any_cm, maj_cm, all_cm), 'test')


if __name__=='__main__':
    if 'WS'=='WS':
        #datadir = '/hdd/juma/data/net_intrusion/ids18/CSVs_r_1.0_m_1.0/WS_l'
        datadir = '/data/juma/data/ids18/CSVs_r_1.0_m_1.0/WS_l'
        evaluator(datadir, 'cnn')
    else:
        csv_dir = 'CSVs_r_1.0_m_1.0'
        #sampler_dirs = ['SFS_SI_9.77_l', 'SGS_e_0.05_l','SRS_SI_10_l','FFS_(8,16,4)_l']
        sampler_dirs = ['SFS_SI_95.33_l', 'SGS_e_1_l','SRS_SI_100_l','FFS_(8,16,40)_l']
        #sampler_dirs = ['SFS_SI_930.75_l','SRS_SI_1000_l','SGS_e_10_l','FFS_(8,16,400)_l']
        sdir = 'SR_1.0'
        for d in sampler_dirs:
            datadir = '/data/juma/data/ids18/{}/{}/{}'.format(csv_dir,sdir,d)
            evaluator(datadir,'cnn')

