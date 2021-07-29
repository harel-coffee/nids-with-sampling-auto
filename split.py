import pandas as pd
import time
import os
from os.path import join
from tqdm import tqdm
import numpy as np
from glob import glob
import ntpath

from multiprocessing import Process

from utils import get_dtype, getSeed, normalize_df
from utils import get_ordered_labels, get_label_to_csv

labels_in_order = get_ordered_labels() # ordered from least frequent to most frequent
label_to_csv = get_label_to_csv()

foldname_regex = '10fold_{}.csv'
NUM_OF_FOLDS = None
K = 10

def chunk_read(csv_file):
    reader = pd.read_csv(csv_file, engine='c', dtype=get_dtype(), chunksize=10**6)
    df = pd.concat([df_chunk for df_chunk in tqdm(reader)], sort=False) # 24 sec 
    return df


def split_n_write_mal(csv_files, label, dataroot):
    print(label)
    df_ls = []
    for csv_filename in csv_files:
            fn = join(dataroot, csv_filename)
            df_i = chunk_read(fn)
            df_i = df_i[df_i['Label']==label]
            df_ls.append(df_i)
    df = pd.concat(df_ls, sort=False)
    print("RAM usage: ",df.memory_usage(deep=True).sum()/(1024*1024*1024 ))
  
    if not label in df.Label.unique():
        return
    assert len(df.Label.unique())==1, "There should be only one label {}".format(df.Label.unique())

    flowids = np.sort(df['Flow ID'].unique())
    num_flows = len(flowids)
    if num_flows<K:
        print("Category {1} has less than K({2}) flows: {0} ".format(num_flows,label,K))
        return
  
    np.random.seed(getSeed())
    np.random.shuffle(flowids) # FLOW shuffle reduces bias in data split while FLOWRECORD shuffle reduces bias in model
    n = num_flows//K
    folds_df = [] 
    for i in range(NUM_OF_FOLDS):
        fn = join(dataroot,foldname_regex.format(i))
        fold_fids = flowids[i*n:(i+1)*n]
        fold_df = df.loc[(df['Flow ID'].isin(fold_fids))].copy()
        fold_df = normalize_df(fold_df)
        folds_df.append(fold_df)
        fsize =  os.path.getsize(fn)
        if fsize==0:
            fold_df.to_csv(fn, index=False)
        else:
            fold_df.to_csv(fn,  mode='a', header=False, index=False)


def open_folds(dataroot):
    for i in range(NUM_OF_FOLDS):
        fn = join(dataroot, foldname_regex.format(i))
        open(fn,'w').close()


def split_malicious(dataroot):
    # Comparison timing for SFS 1/1000 SR, w flow cache of 0.001:
    # debug: 17 sec // release: 7 sec 

    tick = time.time()
    cmds = [(label_to_csv[l],labels_in_order[i], dataroot) for i, l in enumerate(labels_in_order) ]

    open_folds(dataroot)
    for cmd in cmds:
        split_n_write_mal(*cmd)
    print('Malicious split is Done in {:.2f}'.format(time.time()-tick)) # 10min for RPS


def normalize_n_write_normal(df, fn):
    normalize_df(df).to_csv(fn, index=False, chunksize=10**4, mode='a',header=False)


def split_normal(dataroot):
  tick = time.time()
  for csv_file in glob(join(dataroot,'*Meter.csv')):
    df = chunk_read(csv_file)
    df = df.sort_values(['Flow ID']) # cannot shuffle due to approx split
    df = df[df['Label']=='Benign']
   
    flowids = np.sort(df['Flow ID'].unique())  
    np.random.seed(getSeed())
    np.random.shuffle(flowids)
    
    n = len(flowids)//K
    for i in range(NUM_OF_FOLDS):
        fn = join(dataroot,foldname_regex.format(i))
        fids_fold = flowids[i*n:(i+1)*n]
        df_p = df.loc[(df['Flow ID'].isin(fids_fold))].copy()
        normalize_n_write_normal(df_p,fn)
  print("Normal split is done in {:.2f} min ".format((time.time()-tick)/60.))


if __name__=='__main__':
    NUM_OF_FOLDS=10
    if 'noWS'=='WS':
        #datadir = '/hdd/juma/data/net_intrusion/ids18/CSVs_r_0.1/WS_l'
        datadir = '/data/juma/data/ids18/CSVs_r_1.0_m_1.0/WS_l'
        print(datadir)
        split_malicious(datadir)
        split_normal(datadir)
    else:
        csv_dirs = ['CSVs_r_1.0_m_1.0', 'CSVs_r_0.1_m_1.0', 'CSVs_r_0.01_m_1.0', 'CSVs_r_0.001_m_1.0']
        sampler_dirs = ['SFS_SI_9.77_l','SRS_SI_10_l','SGS_e_0.05_l','FFS_(8,16,4)_l']
        #sampler_dirs = ['SFS_SI_95.33_l','SGS_e_1_l','FFS_(8,16,40)_l','SRS_SI_100_l']
        #sampler_dirs = ['SFS_SI_685.08_l','SRS_SI_1000_l','SGS_e_11.5_l','FFS_(8,16,400)_l']
        sr_d = 'SR_10.0'
        for csv_dir in csv_dirs[-1:]:
            for d in sampler_dirs[2:]:  
                datadir = '/data/juma/data/ids18/{}/{}/{}'.format(csv_dir,sr_d,d)
                print(datadir)
                split_malicious(datadir)
                print("**************************")
                split_normal(datadir)
