import pandas as pd
from glob import glob
from os.path import join
import os
from collections import defaultdict
import ntpath
import numpy as np
from multiprocessing import Process


def calc_n_write_flow_observation(flow_dist,dataroot):
    #####################
    root = '/data/juma/data/ids18'
    gt_df = pd.read_csv(join(root,'CSVs_r_1.0_m_1.0/WS_l/flow_dist.csv'),encoding='utf-8',usecols=['Label','Count'],dtype={'Label':str,'Count':int})
    ordered_list = pd.read_csv(join(root,'categories','ordered_list.csv'),header=None)[0].values
    ##########################
    obsr_df = pd.DataFrame(columns=['Label','Count','Observation rate'])
    for label in ordered_list:
        if label in flow_dist.keys():
            gt_count = gt_df[gt_df['Label']==label]['Count'].values[0]
            count = flow_dist[label]
            rate = round(100*count/gt_count,2)
        else:
            count = 0
            rate = 0
        obsr_df = obsr_df.append({'Label':label,'Count':count,'Observation rate':rate},ignore_index=True)

    # we need to obtain (1) average malicious obsr rate and (2) total obsr rate
    ben_obsr_df = obsr_df[obsr_df['Label']=='Benign']
    #print("ben_obsr_df")
    #print(ben_obsr_df)
    m_obsr_rate = (obsr_df['Observation rate'].sum()-ben_obsr_df['Observation rate'].values[0])/(len(ordered_list)-1)
    count_sum = obsr_df['Count'].sum()-ben_obsr_df['Count'].values[0]

    obsr_df = obsr_df.append({'Label':'Macro average Observation Rate','Count':count_sum,'Observation rate':round(m_obsr_rate,2)},ignore_index=True)
    obsr_df.to_csv(join(dataroot,'observation_rate.csv'),index=False,encoding='utf-8-sig')



def process_datadir(dataroot):
    print(dataroot)
    flow_dist = defaultdict(lambda: 0)
    for fn in glob(join(dataroot,'*Meter.csv')):
        print(ntpath.basename(fn))
        df = pd.read_csv(fn,usecols=['Flow ID','Label'],dtype={'Flow ID':str,'Label':str})
        flow_counts = df.groupby(['Label'],as_index=False).agg({'Flow ID':'nunique'})
        for row in flow_counts.iterrows():
            label = row[1]['Label']
            count =row[1]['Flow ID']
            flow_dist[label]+= count
    calc_n_write_flow_observation(flow_dist, dataroot)


if __name__=='__main__':
    sdir = 'SI_10'
    dataroot =   '/data/juma/data/ids18/CSVs_r_1.0_m_1.0/{}'.format(sdir)
    sampling_dirs = [d for d in glob(join(dataroot,'*_l'))  ] 
    print(sampling_dirs)
    procs = [ Process(target=process_datadir, args=[sdir]) for sdir in sampling_dirs]
    for p in procs: p.start()
    for p in procs: p.join()



