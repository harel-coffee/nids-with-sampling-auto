import os
from os.path import join 
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
import glob
import time
import ntpath
import math

from utils import make_value2index, ensure_dir, read_data
from utils import normalize_df, balance_data


def group_data(df, K):
    grouped = df.groupby(['Flow ID','day','Label'], sort=True)
    ID = [ [flowid,label]  for (flowid,label)  in grouped.groups.keys()]
    groupid,count = np.unique(ID,return_counts=True)

    Label = [label for flowid,label in ID]
    ID = np.array(ID)
    return ID,Label


if __name__=='__main__':
    fraction = .01
    file_ending = '*Meter.csv'
    K=5
   
    dataroot = '/data/juma/data/ids18/CSVs/WS_l'
    outputdir = join(dataroot,'folds_fraction_{}'.format(fraction))
    ensure_dir(outputdir)

    df = read_data(dataroot,file_ending, usecols=['Flow ID','Timestamp','Label'])
    df['day'] = df['Timestamp'].map(lambda x: x[:2])
    df['day'] = df['day'].astype(int)

    flowids,flowlabels = group_data(df,K)    
    unique_labels,label_counts = np.unique(flowlabels,return_counts=True)
    flow_observation_rate = np.ones(len(unique_labels))*100
    pd.DataFrame({'Label':unique_labels,'Count':label_counts,'Observation Rate':flow_observation_rate}).to_csv(join(outputdir,'flow_dist.csv'),index=False,encoding='utf-8-sig')

    
    skf = StratifiedKFold(n_splits=K,random_state=SEED)
    for fold_index, (train_index,test_index) in enumerate(skf.split(flowids,flowlabels)):
            print("Fold ",fold_index)
            print("Group IDs shape")
            print(train_index.shape,test_index.shape)
            tick = time.time()
            test_flowids = flowids[test_index]
            fold_df = get_flow_records(test_flowids,df)
            fold_df.to_csv(join(outputdir,'fold_{}.csv'.format(fold_index)),index=False, encoding='utf-8-sig')
