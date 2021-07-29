import pandas as pd
from os.path import join
import numpy as np


id_cols = ['Flow ID', 'Label']

#SL1: check CSVs_r_1.0 and CSVs_r_0.001
#sampler_dir = 'SRS_SI_1000_l'
#sampler_dir = 'SGS_e_1_l'
#sampler_dir = 'SFS_SI_95.33_l'# 
sampler_dir = 'FFS_(8,16,40)_l'

# SL2
#sampler_dir = 'SFS_SI_9.77_l' # fail
#sampler_dir = 'FFS_(8,16,4)_l' # fail
#sampler_dir = 'SRS_SI_10_l'# fail
#sampler_dir = 'SGS_e_0.05_l' # fail



foldname = 'fold_0.csv'
#foldname = 'Friday-16-02-2018_TrafficForML_CICFlowMeter.csv'
sr = 'SR_1.0'
#datadir_rx = '/mnt/disk3/net_intrusion/ids18/{}/SR_1.0/wo_flush/{}/'
#datadir_rx = '/hdd/juma/data/net_intrusion/ids18/{}/{}/{}'
datadir_rx = '/data/juma/data/ids18/{}/{}/{}/'
print(sampler_dir)
print('-------------')
print("{0:20} => ({1: <12},{2: <12}) ".format('CSV_dir','#records','#flows'))
print('-------------')


def get_flows_n_records(datadir, foldname):
    df = pd.read_csv(join(datadir,foldname),usecols=['Flow ID','Label','Timestamp'])
    df['Day'] = df['Timestamp'].map(lambda x: x[:2]).astype(str) # type string
    #rec_df = df[df['Label']!='Benign']
    rec_df = df
    flow_df = rec_df.drop_duplicates(id_cols).sort_values(by=['Flow ID'])
    flow_df = flow_df.sort_values(by=['Flow ID'])
    return rec_df, flow_df     


cache_dirs= ['CSVs_r_1.0_m_1.0', 'CSVs_r_0.1_m_1.0','CSVs_r_0.01_m_1.0', 'CSVs_r_0.001_m_1.0']
#cache_dirs= ['CSVs_r_1.0_m_1.0',  'CSVs_r_0.01_m_1.0']
flows_list = []
for i,csv_dir in enumerate(cache_dirs):
    datadir = datadir_rx.format(csv_dir,sr,sampler_dir)    
    df_rec, df_flow = get_flows_n_records(datadir, foldname)
    print("{0:20} => ({1: <12},{2: <12}) ".format(csv_dir,df_rec.shape[0], df_flow.shape[0]))
    flows_list.append(df_flow['Flow ID'].values)
    df_t = df_flow.sort_values(by=['Day'])
    print(df_flow['Day'].value_counts().sort_index())
    

print('0 vs 1',np.array_equal(flows_list[0],flows_list[1]))
print('0 vs 3',np.array_equal(flows_list[0],flows_list[3]))

print('1 vs 2',np.array_equal(flows_list[1],flows_list[2]))
print('2 vs 3',np.array_equal(flows_list[2],flows_list[3]))
