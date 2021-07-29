import pandas as pd
from os.path import join
from utils import getSeed
import time
from glob import glob
import ntpath
import numpy as np
import subprocess
from utils import get_dtype4normalized
from tqdm import tqdm
from multiprocessing import Pool


def shuffle(fn):
    print(fn)
    df = pd.read_csv(fn, dtype=get_dtype4normalized(), engine='c') # 1min

    df = df.sample(frac=1, random_state = getSeed(), replace=False)# 20 sec

    fn_o = fn.replace('fold','shuffled_fold')
    tick = time.time()
    df.to_csv(fn_o,index=False, chunksize=10**4) # 7min
    print("Wrote in {:.2f}".format(time.time()-tick))


if __name__=='__main__':
    dataroot = '/data/juma/data/ids18/CSVs_r_1.0/SR_10/RPS_SI_10_l'
    regex = join(dataroot,'fold_*.csv')
    fns = [fn for fn in glob(regex) if ntpath.basename(fn) not in ['fold_0.csv','fold_1.csv']]
    p = Pool(len(fns))
    p.map(shuffle,fns)

