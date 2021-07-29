#in pre processing stage we will split (and write) the data with non-overlapping flowids

import numpy as np
from os.path import join as os_join
import os
import csv
from tqdm import tqdm
from collections import defaultdict
from numpy import genfromtxt
from utils import read_csv
#filename = '/home/jumabek/IntrusionDetectionSampling/data/PCAPs/output/Friday-WorkingHours.pcap_Flow.csv'
dataroot = '/home/isrl/data/CIC-IDS-2017/GeneratedLabelledFlows/TrafficLabelling/'

def get_filenames(a_dir):
    return [name for name in os.listdir(a_dir) if os.path.isfile(os.path.join(a_dir,name)) and not name.startswith(".~lock.") and (name.endswith(".pcap_ISCX.csv") or name.endswith(".pcap_Flow.csv"))]


invalid_flow_occurance = 0
total_invalid_flowids = []
def check_flow_occurance(data):
    global invalid_flow_occurance
    invalid_flowids = []
    for key,vals in data.items():
        if len(vals)<2: 
            continue
        vals = np.array(vals)
        if len(set(vals[:,-1]))>0:
            invalid_flow_occurance+= vals.shape[0]
            invalid_flowids.append(key)
            #print(len(set(vals[:,-1])))
            #print(vals[:,0],vals[:,-1],vals[:,6])
    return invalid_flowids

def make_dictionary_from_flow(data):
    d = defaultdict(list)
    for row in data:
        flowid = row[1]+'-'+row[3]+'-'+row[2]+'-'+row[4]+'-'+row[5]
        d[flowid].append(row)
    return d


filenames = get_filenames(dataroot)
print(filenames)
total_flow_occurance = 0
for filename in filenames:
    filename = os.path.join(dataroot,filename)
    data = read_csv(filename)
    total_flow_occurance+=data.shape[0]
    data_d = make_dictionary_from_flow(data)
    invalid_flowids = check_flow_occurance(data_d)
    total_invalid_flowids+=invalid_flowids

with open(os_join(dataroot,"invalid_flowids2.txt"),"w") as f:
    for flowid in total_invalid_flowids:
        f.write("{}\n".format(flowid))

print('Invalid flow  ',invalid_flow_occurance/total_flow_occurance)
