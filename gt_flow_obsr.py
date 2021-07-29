import pandas as pd
from glob import glob
from os.path import join
import os
from collections import defaultdict
import ntpath
import numpy as np
from multiprocessing import Process

def write_flow_dist(flow_dist):
    with open(join(dataroot,'flow_dist.csv'),'w') as f:
        f.write('{},{}\n'.format('Label','Count'))
        for key in sorted(flow_dist.keys()):
            f.write('{},{}\n'.format(key,flow_dist[key]))


#dataroot = '/data/juma/data/ids18/CSVs/WS_l/'
dataroot = '/hdd/juma/data/net_intrusion/ids18/CSVs/WS_l'

flow_dist = defaultdict(lambda: 0)
for i,fn in enumerate(glob(join(dataroot,'*Meter.csv'))):
    print(i,ntpath.basename(fn))
    df = pd.read_csv(fn,usecols=['Flow ID','Label'],dtype={'Flow ID':str,'Label':str})
    #print(df.memory_usage(deep=True).sum()/bytesInGb)

    # flow dist
    flow_counts = df.groupby(['Label'],as_index=False).agg({'Flow ID':'nunique'})
    for row in flow_counts.iterrows():
        label = row[1]['Label']
        count =row[1]['Flow ID']
        flow_dist[label]+= count        


write_flow_dist(flow_dist)




