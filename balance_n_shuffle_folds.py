from os.path import join
from glob import glob
import ntpath
import time 
from multiprocessing import Process

import pandas as pd
import numpy as np

from utils import getSeed
from utils import get_dtype4normalized, get_cols4eval


K=10
foldname_prefix = '{}fold_'.format(K)
def get_count_per_fold(dataroot): # this makes sure every single malicious example is used(almost)
    fn = join(dataroot,'label_dist.csv')
    df = pd.read_csv(fn, names=['Label','Count'], dtype={'Label':str,'Count':int})
    df = df[df['Label']!='Benign']
    cnt = int(df['Count'].max()) # around 100K 
    print("cnt per class = ", cnt)
    return cnt//K


def worker(fn,cnt):
    reader = pd.read_csv(fn, usecols=get_cols4eval(), engine='c', dtype=get_dtype4normalized(),chunksize=10**6) # 1.5 min
    
    df = pd.concat([df for df in reader], sort=False)
    print(ntpath.basename(fn),df.Label.value_counts())
    g = df.groupby(['Label'], sort=False) # 0.00 sec
    new_df = pd.DataFrame(g.apply(lambda x: x.sample(cnt, random_state=getSeed(), replace=True).reset_index(drop=True)))# 33 sec 
    outfile = fn.replace(foldname_prefix,'{}bal_fold_'.format(K))
    new_df = new_df.sample(frac=1, random_state = getSeed(), replace=False) # shuffling, 1min
    tick = time.time()
    new_df.to_csv(outfile,chunksize=10**5, index=False)
    print("Written in {:.2f} ".format(time.time()-tick))# 3.5 mins for SFS

def balance(dataroot): 
    print(ntpath.basename(dataroot))
    cnt = get_count_per_fold(dataroot)
    cmds = [(fold_name,cnt) for fold_name in glob(join(dataroot,'{}*.csv'.format(foldname_prefix)))] 
    for cmd in cmds:
        worker(*cmd) 


if __name__=='__main__':
    if 'noWS'=='WS':
        #datadir = '/hdd/juma/data/net_intrusion/ids18/CSVs/WS_l'
        datadir = '/data/juma/data/ids18/CSVs/WS_l'
        balance(datadir)
    else:
        csv_dir = 'CSVs_r_1.0_m_1.0'
        sampler_dirs = ['SFS_SI_9.77_l', 'SGS_e_0.05_l','SRS_SI_10_l','FFS_(8,16,4)_l']
        #sampler_dirs = ['SFS_SI_95.33_l', 'SGS_e_1_l','FFS_(8,16,40)_l', 'SRS_SI_100_l']
        #sampler_dirs = ['SFS_SI_685.08_l','SRS_SI_1000_l','SGS_e_11.5_l','FFS_(8,16,400)_l']
        sr_d = 'SR_10.0'
        for d in sampler_dirs:
            datadir = '/data/juma/data/ids18/{}/{}/{}'.format(csv_dir,sr_d,d)
            balance(datadir)

