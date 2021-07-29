import pandas as pd
from os.path import join
import numpy as np


id_cols = ['Flow ID', 'Label']
#sampler_dir = 'SRS_SI_100_l'
#sampler_dir = 'SGS_e_1_l'
#sampler_dir = 'SFS_SI_930.75_l'
#sampler_dir = 'FFS_(8,16,40)_l'

sampler_dir = 'SFS_SI_9.77_l' # 
#sampler_dir = 'FFS_(8,16,4)_l' # 
#sampler_dir = 'SRS_SI_10_l'# 
#sampler_dir = 'SGS_e_0.05_l' # 

csv_dir = 'CSVs_r_1.0_m_1.0'
sdir = 'SR_10.0' 
datadir = '/data/juma/data/ids18/{}/{}/{}/'.format(csv_dir,sdir,sampler_dir)
for i in range(5):
    fn = join(datadir,'fold_{}.csv'.format(i))
    print('---------fold #{}------------'.format(i))
    print(pd.read_csv(fn, usecols=['Label'])['Label'].value_counts())

