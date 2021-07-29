import pandas as pd
from os.path import join
import numpy as np
import ntpath
from glob import glob

id_cols = ['Flow ID', 'Label']
dtypes = {'Flow ID':str,'Label': str}
#sampler_dir = 'SRS_SI_100_l'
#sampler_dir = 'SGS_e_1_l'
#sampler_dir = 'SFS_SI_930.75_l'
#sampler_dir = 'FFS_(8,16,40)_l'

# for 1/10: CSVs_r_1.0_m_1.0
#sampler_dir = 'SFS_SI_9.77_l'   # pass 
sampler_dir = 'FFS_(8,16,4)_l' # pass 
#sampler_dir = 'SRS_SI_10_l'    # pass 
#sampler_dir = 'SGS_e_0.05_l'   # pass

foldnames = ['fold_0.csv', 'vfold_0.csv']

sdir = 'SR_10.0' 
datadir = '/data/juma/data/ids18/CSVs_r_1.0_m_1.0/{}/{}/'.format(sdir,sampler_dir)
if 1==1:
    
    fn1 = join(datadir, foldnames[0])
    fn2 = join(datadir, foldnames[1])

    df1 = pd.read_csv(fn1,usecols = id_cols, dtype= dtypes)
    df2 = pd.read_csv(fn2, usecols = id_cols, dtype = dtypes)
    print('# records:\n', df1.shape[0],'\n', df2.shape[0])

    df1 = df1.sort_values(['Flow ID'])
    df2 = df2.sort_values(['Flow ID'])

    fid1 = np.sort(df1['Flow ID'].unique())
    fid2 = np.sort(df2['Flow ID'].unique())
    print("same flows? = ", np.array_equal(fid1, fid2))

