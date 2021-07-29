import os
from os.path import join 
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pickle

import argparse
import pandas as pd
import glob
import time
import ntpath

from utils import make_value2index, ensure_dir, read_data
from utils import normalize_df, balance_data
from utils import getSeed
import math


def round_up(n, decimals=0):
    multiplier = 10 ** decimals
    return math.ceil(n * multiplier) / multiplier


def group_data(df, K, outputdir):
    #remove classes less than K items
    print("Grouping by FlowID and Label")
    labels = [ label for (flowid,label) in df.groupby(['Flow ID','Label'],sort=True).groups.keys()]
    calc_n_write_flow_observation(labels,outputdir)
    print("logged flow observation rate")    
 
    unique,count = np.unique(labels,return_counts=True)

    print('-----------------------------------')
    print("REMOVING VERY SMALL CLASSES")
    for label,count in zip(unique,count):
        if count<K:
            df = df[df['Label']!=label]
            print(label)
    print('-----------------------------------')

    # after deleting small classes regroup again
    print("Re-Grouping by FlowID and Label")
    grouped = df.groupby(['Flow ID','Label'],sort=True)
    ID = [ [flowid,label]  for (flowid,label)  in grouped.groups.keys()]
    groupid,count = np.unique(ID,return_counts=True)

    Label = [label for flowid,label in ID]
    ID = np.array(ID)
    return ID,Label, grouped


def get_flow_records(ids, df,grouped):
    frames_list = [grouped.get_group((flowid,label)) for flowid,label in ids]
    print("Concatinating obtained frames")
    df = pd.concat(frames_list,sort=False)
    return df


def calc_n_write_flow_observation(flowlabels,outputdir):
    #####################
    gt_df = pd.read_csv(join(root,'CSVs/WS_l/folds_fraction_1/flow_dist.csv'),encoding='utf-8',usecols=['Label','Count'],dtype={'Label':str,'Count':int})
    benign_list = pd.read_csv(join(root,'benign_list.csv'),header=None)[0].values
    short_attack_list = pd.read_csv(join(root,'short_attack_list.csv'),header=None)[0].values
    long_attack_list = pd.read_csv(join(root,'long_attack_list.csv'),header=None)[0].values
    ##########################
 
    unique_labels,label_counts = np.unique(flowlabels,return_counts=True)
    observation_df = pd.DataFrame(columns=['Label','Count','Observation rate'])
    macro_avg = 0
    # benign
    rates = 0
    counts = 0
    for label in benign_list:
        if label in unique_labels:
            index= np.where(label==unique_labels)[0][0]
            gt_count = gt_df[gt_df['Label']==label]['Count'].values[0]
            count = label_counts[index]
            rate = round_up(100*count/gt_count,2)
        else:
            count = 0
            rate = 0
        observation_df = observation_df.append({'Label':label,'Count':count,'Observation rate':rate},ignore_index=True)

    observation_rate_sum = 0
    count_sum = 0
    # short attacks
    for label in short_attack_list:
        if label in unique_labels:
            index= np.where(label==unique_labels)[0][0]
            gt_count = gt_df[gt_df['Label']==label]['Count'].values[0]
            count = label_counts[index]
            rate = round_up(100*count/gt_count,2)
        else:
            count = 0
            rate = 0
        observation_rate_sum+=rate
        count_sum +=count
        observation_df = observation_df.append({'Label':label,'Count':count,'Observation rate':rate},ignore_index=True)

    # long attacks
    for label in long_attack_list:
        if label in unique_labels:
            index= np.where(label==unique_labels)[0][0]
            gt_count = gt_df[gt_df['Label']==label]['Count'].values[0]
            count = label_counts[index]
            rate = round_up(100*count/gt_count,2)
        else:
            count = 0
            rate = 0
        observation_df = observation_df.append({'Label':label,'Count':count,'Observation rate':rate},ignore_index=True)
        observation_rate_sum+=rate
        count_sum +=count

    m_observation_rate = observation_rate_sum/(len(short_attack_list)+len(long_attack_list))
    observation_df = observation_df.append({'Label':'Macro average Observation Rate','Count':count_sum,'Observation rate':m_observation_rate},ignore_index=True)
    observation_df.to_csv(join(outputdir,'observation_rate.csv'),index=False,encoding='utf-8-sig')


#root = '/mnt/sda_dir/juma/data/net_intrusion/CIC-IDS-2018/'
root = '/data/juma/data/ids18'
def make_fold(dataroot):
    fraction = 1
    file_ending = '*Meter.csv'
    K=5
   
    outputdir = join(dataroot,'folds_fraction_{}'.format(fraction))
    ensure_dir(outputdir)

    df = read_data(dataroot,file_ending,fraction=fraction)
    df = normalize_df(df,join(outputdir,'data_stats.pickle'),train_data=True)
    flowids,flowlabels,grouped = group_data(df,K, outputdir)    
   
    skf = StratifiedKFold(n_splits=K,shuffle=True, random_state=getSeed())
    for fold_index, (train_index,test_index) in enumerate(skf.split(flowids,flowlabels)):
            print("Fold - ",fold_index)
            test_flowids = flowids[test_index]
            fold_df = get_flow_records(test_flowids,df,grouped)
            fold_df.to_csv(join(outputdir,'fold_{}.csv'.format(fold_index)),index=False, encoding='utf-8-sig')


if __name__=='__main__':
    #dataroot = '/data/juma/data/ids18/CSVs/WS_l'
    dataroot = '/hdd/juma/data/net_intrusion/ids18/CSVs/WS_l'
    make_fold(dataroot)
