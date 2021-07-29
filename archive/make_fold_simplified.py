import os
from os.path import join 
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
from glob import glob
import time
import ntpath
from tqdm import tqdm

from utils import normalize_df, balance_data
from utils import get_dtype, getSeed


def get_flow_records_by_gid(ids,df):
    cols = ['Flow ID', 'Label']
    grouped = df.groupby(by=cols)
    chunk_gids = list(grouped.groups.keys())
    intersect_ids = set(chunk_gids).intersection(set(ids))
    if len(intersect_ids)==0:
        return None 
    print("\nchunk GID/Total GID/intersectdGID {}/{}/{}".format(len(chunk_gids),len(ids),len(intersect_ids)))
    #records = grouped.filter(lambda x: (x['Flow ID'], x['Day'], x['Label']) in ids )
    records = pd.concat([grouped.get_group(id) for id in tqdm(intersect_ids)])
    #print(records.shape)
    return records 


def get_flowids_and_labels(df):
    cols = ['Flow ID','Label']
    df_id = df[cols].drop_duplicates()

    flowlabels = df_id['Label'].values
    gids = list(df_id.groupby(cols, sort=False).groups.keys())
    print('# groups = ',len(gids))
    return gids,flowlabels


def make_fold_i(dataroot,flowids,fold_index):
    #filter data based on flowids
    #chunksize 10^4: 100min
    # chunksize 10^5: 12min
    # chunksize 10^6: 3min
    chunksize = 10**6
    
    df_list = []
    for i,fn in enumerate(glob(join(dataroot,'*Meter.csv'))):
        tick_start = time.time()
        TextFileReaderObject = pd.read_csv(fn,engine='c',dtype=get_dtype(),chunksize=chunksize)
        df_per_file = pd.concat([df_chunk[df_chunk['Label']!='Benign'] for df_chunk in tqdm(TextFileReaderObject)], sort=False)
        print("CSV file per day is readNconcat in {:.2f}".format(time.time()-tick_start))
        tick_record = time.time()
        df_per_file = get_flow_records_by_gid(flowids,df_per_file)
        if df_per_file is None: # no flow from this CSV for the given fold
            continue
        print('Flow records are obtain in {:.2f} sec'.format(time.time()-tick_record))

        tick = time.time()
        print("df_per_file.shape = ", df_per_file.shape)
        df_per_file_norm = normalize_df(df_per_file)
        print('normalalization time {:.2f} sec'.format(time.time()-tick))
        print(df_per_file.shape)
        if i==0:
            df_per_file_norm.to_csv(join(dataroot,'fold_{}.csv'.format(fold_index)))
            df_per_file.to_csv(join(dataroot,'nonnormalized_fold_{}.csv'.format(fold_index)))
        else:
            df_per_file_norm.to_csv(join(dataroot,'fold_{}.csv'.format(fold_index)),mode='a',header=False)
            df_per_file.to_csv(join(dataroot,'nonnormalized_fold_{}.csv'.format(fold_index)),mode='a',header=False)
        print("Time spent per CSV file {:.2f} ".format(time.time() - tick_start))
    print("Done for fold ",fold_index)


def caller(dataroot):
    K=5
    #dataroot = '/data/juma/data/ids18/CSVs/WS_l_old'
    dataroot = '/data/juma/data/ids18/CSVs_r_1.0/SR_10/RPS_SI_10_l'
    print(ntpath.basename(dataroot))

    # loading flow ids, takes 100 sec
    tick = time.time()
    df_list = []
    for i,fn in enumerate(tqdm(glob(join(dataroot,'*Meter.csv')))):
        df = pd.read_csv(fn, usecols=['Flow ID','Label'],dtype={'Flow ID':str,'Label':str}) #20min, for RPS_10: 
        # 1. drop duplicates 2.filter out.
        df = df.drop_duplicates(subset=['Flow ID','Label'])
        df = df[df['Label']!='Benign']
        df_list.append(df)
    df = pd.concat(df_list,sort=False)
    print("Flow ids are read in {:.2f} sec".format(time.time()-tick)) # data is read in 6 min, 2 min for RPS_10

    tick = time.time()
    flowids, flowlabels = get_flowids_and_labels(df)
    tock = time.time()
    print('obtained UNIQUE flowid and labels in {:.2f} sec'.format(tock-tick)) # 1100sec for RPS
    skf = StratifiedKFold(n_splits=K,random_state=getSeed(), shuffle=True)
    tick = time.time()
    flowids_per_fold = []
    for fold_index, (train_index,test_index) in enumerate(skf.split(np.zeros(len(flowlabels)),flowlabels)):
        if fold_index>=3:
            tock = time.time()
            print("-------------{}---------------Kfold split took: {:.2f} sec".format(fold_index,tock-tick))
            tick = time.time()
            test_flowids = [flowids[i] for i in test_index]
            unique, counts = np.unique(flowlabels[test_index], return_counts=True)
            #print("Testing fold ", fold_index, get_overlap(test_flowids,flowids[train_index]))
            make_fold_i(dataroot,test_flowids, fold_index)
            print("Fold #{} is done in {:.2f}".format(fold_index, time.time()-tick))
            
if __name__=='__main__':
    dataroots = [
        '/data/juma/data/ids18/CSVs_r_1.0/SR_10/RPS_SI_10_l',
        '/data/juma/data/ids18/CSVs_r_1.0/SR_10/FFS_(8,16,4)_l',
        '/data/juma/data/ids18/CSVs_r_1.0/SR_10/SFS_SI_9.77_l',
        '/data/juma/data/ids18/CSVs_r_1.0/SR_10/SGS_e_0.000107_l',
        '/data/juma/data/ids18/CSVs/WS_l'
    ]
    for dataroot in dataroots:
        caller(dataroot)
        break
