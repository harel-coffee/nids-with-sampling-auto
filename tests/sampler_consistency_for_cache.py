import pandas as pd
from os.path import join
import numpy as np
import ntpath
from glob import glob
import sys


id_cols = ['Flow ID', 'Label']
dtypes = {'Flow ID':str,'Label': str}

# I check SR_1.0// Sep 25// r_0.1 and r_0.01 
# II: check fold_0.csv
#sampler_dir = 'SRS_SI_100_l'# pass, pass
#sampler_dir = 'SGS_e_1_l' # pass , pass
#sampler_dir = 'SFS_SI_95.33_l' #pass, pass
#sampler_dir = 'FFS_(8,16,40)_l' #pass, pass

#beforie, after sampled on Tuesday of Septermber 21
#sampler_dir = 'SFS_SI_9.77_l'   #fail,pass 
#sampler_dir = 'FFS_(8,16,4)_l' # fail, pass 
#sampler_dir = 'SRS_SI_10_l'    # fail, pass 
#sampler_dir = 'SGS_e_0.05_l'   # pass 

csv = 'CSVs_r_1.0_m_1.0'
csv1 = 'CSVs_r_0.1_m_1.0'
csv2 = 'CSVs_r_0.01_m_1.0'
csv3 = 'CSVs_r_0.001_m_1.0'

sdir = 'SR_1.0' 
regex = '/data/juma/data/ids18/{}/{}/{}/'
datadir = regex.format(csv, sdir, sampler_dir)

print(sampler_dir)

for fn in glob(join(datadir,'fold_0.csv')):
#day = 'Friday-16-02-2018_TrafficForML_CICFlowMeter.csv'
#day = 'Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv' 
    fn1 = fn.replace(csv, csv1)
    fn2 = fn.replace(csv, csv2)
    fn3 = fn.replace(csv, csv3)
    try:
        df = pd.read_csv(fn,usecols = id_cols, dtype= dtypes)
        #print('r_1:',df.shape)
        df1 = pd.read_csv(fn1, usecols = id_cols, dtype = dtypes)
        #print('r_0.1:',df1.shape)
        df2 = pd.read_csv(fn2, usecols = id_cols, dtype = dtypes)
        #print('r_0.01:',df2.shape)
        df3 = pd.read_csv(fn3, usecols = id_cols, dtype = dtypes)
        #print('r_0.001:',df3.shape)
    except:
        print("Probably reading error:", sys.exc_info()[0])
        exit()
    df = df.sort_values(['Flow ID'])
    df1 = df1.sort_values(['Flow ID'])
    df2 = df2.sort_values(['Flow ID'])
    df3 = df3.sort_values(['Flow ID'])

    fid = np.sort(df['Flow ID'].unique())
    fid1 = np.sort(df1['Flow ID'].unique())
    fid2 = np.sort(df2['Flow ID'].unique())
    fid3 = np.sort(df3['Flow ID'].unique())

    is_same_num_flows =  np.array_equal(fid, fid1) and np.array_equal(fid2, fid3) \
    and np.array_equal(fid1, fid2)
  
    if not is_same_num_flows: 
        print('Problem in --------{}----------'.format(ntpath.basename(fn1)[:20]))
        
        print('# flows:\n', fid.shape[0],fid1.shape[0],'\n', fid2.shape[0], fid3.shape[0])

