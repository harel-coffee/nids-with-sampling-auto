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


SEED = 234
def get_flow_records(ids,df):
    tick = time.time()
    new_df = df[(df['Flow ID'].isin(ids[:,0])) & (df['Day'].isin(ids[:,1])) & (df['Label'].isin(ids[:,2]))]
    tock = time.time()
    print('get_records in {:.2f} sec'.format(tock-tick))
    return new_df  
   

def group_data(df, K):
    tick = time.time()
    grouped = df.groupby(['Flow ID','Day','Label'],sort=True)
    tock = time.time()
    print("Grouping done in {:.2f} sec".format(tock-tick))
    tick = time.time()
    ID= np.array(list(grouped.groups.keys()))
    tock=time.time()
    print("converted to array in {:.2f} sec".format(tock-tick))
    # sorting is necessary for reproducing, can be removed for performance if you do not  need same record generation each time
    tick= time.time()
    ID = ID[ID[:,2].argsort()]
    ID = ID[ID[:,1].argsort()]
    ID = ID[ID[:,0].argsort()]
    tock = time.time()
    print("sorted IDs in {:.2f}".format(tock-tick))

    Label = ID[:,2] 
    return ID,Label, grouped


def get_flowids_and_labels(df):
    df_id = df[['Flow ID','Day','Label']]
    df_id = df_id.drop_duplicates(keep='first')
    flowlabels = df_id['Label'].values
    flowids= df_id.values
    #print(labels_per_group.shape, flowids.shape)
    return flowids,flowlabels



if __name__=='__main__':
    K=5
   
    #dataroot = '/data/juma/data/ids18/CSVs/WS_l'
    dataroot = '/hdd/juma/data/net_intrusion/ids18/CSVs/WS_l'
    outputdir = join(dataroot)
    #ensure_dir(outputdir)

    nrows=None
    print('nrows = {} '.format(nrows),end=':\n ')
    tick = time.time()
    df = read_data(dataroot, nrows=nrows) #20min
    print("Data is read in {:.2f} sec".format(time.time()-tick))

    tick = time.time()
    df['Day'] = df['Timestamp'].map(lambda x: x[:2]).astype(str) # type string
    print('new column created in {:.2f} sec'.format(time.time()-tick))
 
    tick = time.time()
    df = normalize_df(df,join(outputdir,'data_stats.pickle'),train_data=False)
    print("Done normalizing in {:.2f} sec".format(time.time()-tick))

    tick = time.time()
    flowids, flowlabels = get_flowids_and_labels(df)
    tock = time.time()
    print('obtained flowid and labels in {:.2f} sec'.format(tock-tick))

    unique, counts = np.unique(flowlabels, return_counts=True)
    print (np.asarray((unique, counts)).T)

    skf = StratifiedKFold(n_splits=K,random_state=SEED)
    tick = time.time()
    for fold_index, (train_index,test_index) in enumerate(skf.split(flowids,flowlabels)):
            tock = time.time()
            print("----------------------------Kfold split took: {:.2f} sec".format(tock-tick))
            tick = time.time()
            test_flowids = flowids[test_index]
            fold_df = get_flow_records(test_flowids, df)
            tick = time.time()
            fold_df.to_csv(join(outputdir,'fold_{}.csv'.format(fold_index)),index=False, encoding='utf-8-sig')
            print("Fold is written in {:.2f}".format(time.time()-tick))
            tick = time.time()

