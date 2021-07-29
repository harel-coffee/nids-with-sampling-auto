import numpy as np
from collections import defaultdict
import os
from os.path import join as os_join
from utils import get_complete_label_names,read_csv
from tqdm import tqdm

dataroot = '/home/isrl/data/CIC-IDS-2017/GeneratedLabelledFlows/TrafficLabelling'


def make_dictionary_from_flow(data):
    d = defaultdict(list)
    print('making dictionary from flow')
    for row in tqdm(data):
        flowid = row[1]+'-'+row[2]+'-'+row[3]+'-'+row[4]+'-'+row[5]  
        d[flowid].append(row) 
    return d


def read_data(dataroot):
    filenames = get_filenames(dataroot)
    data = read_csv(os_join(dataroot,filenames[0]))
    for filename in filenames[1:]:
       data_i = read_csv(os_join(dataroot,filename))
       data = np.concatenate((data,data_i),axis=0)
    return data


def get_filenames(a_dir):
    return [name for name in os.listdir(a_dir) if os.path.isfile(os.path.join(a_dir,name)) and not name.startswith(".~lock.") and (name.endswith(".pcap_ISCX.csv") or name.endswith(".pcap_Flow.csv"))]


label_names = get_complete_label_names()

output_filename = os_join(dataroot,"percentiles_for_num_of_packets_forward.csv")   
if os.path.exists(output_filename):
    os.remove(output_filename)

filenames = get_filenames(dataroot)
flow_lengths = defaultdict(list)

for filename in filenames: # E: Friday-WorkingHours.pcap_Flow.csv
    data = read_csv(os_join(dataroot,filename))
    
    for i in range(data.shape[0]):
        row = data[i]
        flowid = row[1]+'-'+row[2]+'-'+row[3]+'-'+row[4]+'-'+row[5]
        forward_packets = int(row[8])
        flow_label = row[-1]        
        flow_lengths[flow_label].append(forward_packets)

percentile = np.array(['Label','num_flows','Min','90-th percentile','95-th percentile','99-th percentile','99.9-th percentile','100-th percentile'])
percentile = percentile.reshape((1,-1))
for label in label_names:  
   flow_lengths_for_label = flow_lengths[label]
   print("{:40s}-->{:10d}".format(label,len(flow_lengths_for_label)))
   if len(flow_lengths_for_label)<1:
       continue
   flow_lengths_for_label = np.array(flow_lengths_for_label)
  
   row = np.array([label,len(flow_lengths_for_label),np.min(flow_lengths_for_label),\
           np.percentile(flow_lengths_for_label,90),np.percentile(flow_lengths_for_label,95),np.percentile(flow_lengths_for_label,99),\
           np.percentile(flow_lengths_for_label,99.9), np.percentile(flow_lengths_for_label,100)])
   percentile = np.concatenate((percentile,row.reshape((1,-1))),axis=0)
np.savetxt(output_filename,percentile,fmt='%s',delimiter=',')

for key,vals in flow_lengths.items():
    print("len(flow_lengths[{:50}]) = {:10d}".format(key,len(vals)))
