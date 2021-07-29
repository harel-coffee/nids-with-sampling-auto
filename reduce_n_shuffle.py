'''
Purpose is to get equal amount of data compared to oversampling based balancing. We do this by excluding EXTRA benign records
'''

from glob import glob
import pandas as pd
from os.path import join
from tqdm import tqdm
import time

from utils import getSeed, get_dtype4normalized

in_fold = 'fold_'
out_fold = 'r_fold_'
def get_bmd(dataroot): # this makes sure every single malicious example is used(almost)
    fn = join(dataroot,'label_dist.csv')
    df = pd.read_csv(fn, names=['Label','Count'], dtype={'Label':str,'Count':int})
    b = df[df['Label']=='Benign']['Count'].values[0]
    
    df = df[df['Label']!='Benign']
    m = df['Count'].max()
    d = (m - df['Count']).sum() # duplicated malicious count in over sample balancing
    return b,m,d  



# 
# WS: reduce data 5x by reducing benign to 10%
#largest attack category has 4.1% of the benign category
benign_keep_ratio = .5
def fold_worker(fn, b,m,d):
    #Assumptions:
    # 1. Benign records are in the beginning of each fold
    # 2. num of benign records is `b`
    # 3. num of duplicates made in over/under sampling is `d`
    # 4. num of malicious duplicated records  `m`

    # then, using `m+d` benign records ensures we have equal #records against balancing case
    # which means we should remove last `b-(m+d)` benign records from fold
    K=5 # num iof folds
    num_to_exclude = (b - (m+d))//K
     

    tick = time.time()    
    df = pd.read_csv(fn, engine='c', dtype=get_dtype4normalized()) # 4~5 min
    print("Read fold in {:.2f} min".format((time.time()-tick)/60))
    N = df.shape[0]
    r_df = df[:N-num_to_exclude]
    
    sh_df = r_df.sample(frac=1,random_state=getSeed(), replace=False)

    outfile = fn.replace(in_fold, out_fold)
    assert fn!=outfile, "outfile is same as input file {}".format(ntpath.basename(fn))
    sh_df.to_csv(outfile,chunksize=10**5, index=False)

def reduce_n_shuffle(dataroot):
    
    b, m, d = get_bmd(dataroot)
    cmds = [(fold_name, b, m, d) for fold_name in glob(join(dataroot,'fold*.csv')) if 'fold_2' not in fold_name and 'fold_0' not in fold_name]

    for cmd in tqdm(cmds): # RPS:11min 
         fold_worker(*cmd)


if __name__=='__main__':
    dataroot = '/data/juma/data/ids18/CSVs_r_1.0_m_1.0/WS_l'
    #dataroot = '/data/juma/data/ids18/CSVs_r_1.0/SR_10/RPS_SI_10_l'
    reduce_n_shuffle(dataroot)
