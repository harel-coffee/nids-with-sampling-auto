import os.path
from os.path import join as os_join
import csv
from numpy import genfromtxt
import numpy as np
import pandas as pd

dataroot = '/hdd/juma/data/net_intrusion/CIC-IDS-2018/CSVs/sk_sr_1.0_l'

def get_filenames(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isfile(os.path.join(a_dir, name)) and not name.startswith(".~lock.") and (name.endswith('pcap_Flow.csv') or name.endswith('_CICFlowMeter.csv'))]


def read_label_from_csv(filename):
    convertfunc = lambda x: x.strip('"')
    with open(filename,encoding='utf-8') as csv_file:
        print('Reading {} ...'.format(filename))
        data = None
        try:
            data = pd.read_csv(filename)
        except (Exception, e):
            print('errror occured with genfromtxt')
            exit()
        data.columns = data.columns.str.lstrip()
        print(data.shape)
        protocols=[1,6,17]
        
        #data.drop_duplicates(subset=['Source IP','Source Port', 'Destination IP','Destination Port','Protocol'],keep='first',inplace=True)
        print(data.columns)
        print(data.shape)
        #data = data[data.Protocol.isin(protocols)]
    return data['Label'].values


filenames = get_filenames(dataroot)
labels = np.empty((0,))
for filename in filenames:
    output = read_label_from_csv(os_join(dataroot,filename))
    print(output.shape)
    labels= np.concatenate((labels,output),axis=0)

unique, counts = np.unique(labels, return_counts=True)
print("Unique elments: ", len(unique))
distribution = np.asarray((unique, counts)).T
print(distribution)
np.savetxt(os_join(dataroot,'label_dist.csv'),distribution,fmt='%s',delimiter=',',encoding='utf-8-sig')
