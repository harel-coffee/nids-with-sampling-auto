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


def get_flow_records(ids,df):
    #make a Day column
    df['Day'] = df['Timestamp'].map(lambda x: x[:2]).astype(str) # type string

    tick = time.time()
    new_df = df[(df['Flow ID'].isin(ids[:,0])) & (df['Day'].isin(ids[:,1])) & (df['Label'].isin(ids[:,2]))]
    tock = time.time()
    #print('get_records in {:.2f} sec'.format(tock-tick))
    return new_df  


def get_flow_records_loop(ids,df):
    #make a Day column
    df['Day'] = df['Timestamp'].map(lambda x: x[:2]).astype(str) # type string

    tick = time.time()
    df_list = []
    for flowid,day,label in tqdm(ids):
        df_list.append(df[(df['Flow ID']==flowid) & (df['Day']==day) & (df['Label']==label)])
    #new_df = df[(df['Flow ID'].isin(ids[:,0])) & (df['Day'].isin(ids[:,1])) & (df['Label'].isin(ids[:,2]))]
    
    tock = time.time()
    #print('get_records in {:.2f} sec'.format(tock-tick))
    return pd.concat(df_list,sort=False)  


def get_flow_records_by_gid(ids,df):
    cols = ['Flow ID', 'Day','Label']
    df['Day'] = df['Timestamp'].map(lambda x: x[:2]).astype(str) # type string
    grouped = df.groupby(by=cols)
    chunk_gids = list(grouped.groups.keys())
    intersect_ids = set(chunk_gids).intersection(set(ids))
    print("\nchunk GID/Total GID/intersectdGID {}/{}/{}".format(len(chunk_gids),len(ids),len(intersect_ids)))
    #records = grouped.filter(lambda x: (x['Flow ID'], x['Day'], x['Label']) in ids )
    records = pd.concat([grouped.get_group(id) for id in tqdm(intersect_ids)])
    #print(records.shape)
    return records 


def get_flowids_and_labels(df):
    tick = time.time()
    df['Day'] = df['Timestamp'].map(lambda x: x[:2]).astype(str) # type string
    print('new column created in {:.2f} sec'.format(time.time()-tick))
    df = df.drop(columns=['Timestamp'])
 
    cols = ['Flow ID','Day','Label']
    df_id = df[cols].drop_duplicates()
    #print(df_id.Label.value_counts())    

    flowlabels = df_id['Label'].values
    #gids= df_id.values
    print("")
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
        df_per_file = pd.concat(get_flow_records_by_gid(flowids,df_chunk) for df_chunk in tqdm(TextFileReaderObject))
        print('Flow records are obtain in {:.2f} sec'.format(time.time()-tick_start))

        tick = time.time()
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


def get_overlap(test_flowid, train_flowid):
        # assume flowid has two cols: flowid, label
        overlap = set(test_flowid[:,0]).intersection(set(train_flowid[:,0]))
        print("# overlap flowids samples ",len(overlap))
        mask = np.isin(test_flowid[:,0],list(overlap))   
        ovlp_test_flowid = test_flowids[mask]

        train_mask = np.isin(train_flowid[:,0],list(overlap))
        ovlp_train_flowid = train_flowid[train_mask]
        ovlp_by_label = 0
        ovlp_by_labelnday = 0
        labels = []
        for (flowid,day,label) in tqdm(ovlp_test_flowid):
            train_gid = ovlp_train_flowid[ovlp_train_flowid[:,0]==flowid]
            
            if label in train_gid[:,2]:
                if day in train_gid[:,1]:
                    if label!='Benign':
                        print("FULL overlap for the malicious category")
                        print((flowid,day,label))
                    ovlp_by_labelnday+=1
                else:
                    print((flowid,day,label), train_gid[:,2]) 
                    labels.append(train_gid[:,2])
                    ovlp_by_label+=1
        print("Overlaps by label and labelnday ", ovlp_by_label, ovlp_by_labelnday)
        print(np.unique(labels))
        return len(overlap)


def caller(dataroot):
    K=5
    #dataroot = '/data/juma/data/ids18/CSVs/WS_l_old'
    dataroot = '/data/juma/data/ids18/CSVs_r_1.0/SR_10/RPS_SI_10_l'
    print(ntpath.basename(dataroot))

    # loading flow ids, takes 100 sec
    tick = time.time()
    df_list = []
    for i,fn in enumerate(tqdm(glob(join(dataroot,'*Meter.csv')))):
        df = pd.read_csv(fn, usecols=['Flow ID','Timestamp','Label'],dtype={'Flow ID':str,'Timestamp':str,'Label':str}) #20min, for RPS_10: 
        df_list.append(df.drop_duplicates(subset=['Flow ID','Label']))
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
            tock = time.time()
            print("-------------{}---------------Kfold split took: {:.2f} sec".format(fold_index,tock-tick))
            tick = time.time()
            test_flowids = [flowids[i] for i in test_index]
            #test_flowids = flowids[test_index]
            #print(flowids.shape, test_flowids.shape)
            unique, counts = np.unique(flowlabels[test_index], return_counts=True)
            #print("Testing fold ", fold_index, get_overlap(test_flowids,flowids[train_index]))
            make_fold_i(dataroot,test_flowids, fold_index)


if __name__=='__main__':
    dataroots = [
        '/data/juma/data/ids18/CSVs_r_1.0/SR_10/FFS_(8,16,4)_l',
        '/data/juma/data/ids18/CSVs_r_1.0/SR_10/SFS_SI_9.77_l',
        '/data/juma/data/ids18/CSVs_r_1.0/SR_10/SGS_e_0.000107_l',
        '/data/juma/data/ids18/CSVs/WS_l'
    ]
    for dataroot in dataroots:
        caller(dataroot)
