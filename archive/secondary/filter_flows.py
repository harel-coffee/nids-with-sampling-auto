import numpy as np
from os.path import join as os_join
import os
import csv
from tqdm import tqdm
from collections import defaultdict
from numpy import genfromtxt
from utils import read_csv, read_csv_header
import time

dataroot = '/home/isrl/data/CIC-IDS-2017/sketchflow'
#good_flow_dataroot = '/home/isrl/data/CIC-IDS-2017/PCAPs/output'
good_flow_dataroot = '/home/isrl/data/CIC-IDS-2017/GeneratedLabelledFlows/TrafficLabellingMerged'

output_dataroot = dataroot+'_filtered'
if not os.path.exists(output_dataroot):
    os.mkdir(output_dataroot)

flowid_indices = [0,1,2,3,4,5,6] # Flow identification, although i think we do not need timestamp

ID=1
SR=10
feature_indices=      [8 ,9 ,10,11,12,13,14,15,16,17,18,19,40,41,44,45,46,47,48,57,58,59,60,73,74]
#factor_for_features =[SR,SR,SR,SR,ID,ID,ID,ID,ID,ID,ID,ID,SR,SR,ID,ID,ID,ID,ID,ID,ID,ID,ID,SR,ID]
important_indices = flowid_indices + feature_indices


def get_filenames(a_dir):
    return [name for name in os.listdir(a_dir) if os.path.isfile(os.path.join(a_dir,name)) and not name.startswith(".~lock.") and (name.endswith(".pcap_ISCX.csv") or name.endswith(".pcap_Flow.csv"))]


def make_dictionary_from_flow(data):
    d = defaultdict(list)
    print('making dictionary from a flow')
    for row in tqdm(data):
        flowid = row[1]+'-'+row[2]+'-'+row[3]+'-'+row[4]+'-'+row[5]  
        d[flowid].append(row) # row[0] is flow id:x
    print("length of the dictionary made: ",len(d))
    return d


def extract_n_fix(data_d,good_flowids): # fixes some feat:xuers that need to be muliplied by smaplfing rate
    print('Extracting good flows and important features')
    data = []
    missed_good_flows = 0
    zero_subflows = 0
    total_forward_packets_id = 8
    total_backward_packets_id = 9
    for flowid in tqdm(good_flowids):
        flow = data_d[flowid]
        flow = np.array(flow)
        if flow.shape[0]<1:
            missed_good_flows+=1
            continue        
        if 'output_filtered' not in output_dataroot and (int(flow[0][total_forward_packets_id])<1 or int(flow[0][total_backward_packets_id])<1):
            zero_subflows+=1
            continue
        if len(set(flow[:,-1]))>1:
            print(flow)
            print('This should not happen, because {} is good_flowid'.format(flowid))
            exit(1)
        for i in range(flow.shape[0]):
            data.append(flow[i])
    data = np.array(data)
    print(data.shape)
    if 'output_filtered' not in output_dataroot:
        data[:,[8,9,10,11,40,41,73]] = np.array(data[:,[8,9,10,11,40,41,73]].astype(float)*SR).astype(int).astype(str)
    data = data[:,important_indices] 
    print("missed good flows: ", missed_good_flows)
    print("zero subflows: ", zero_subflows)
    return data


filenames  = get_filenames(dataroot)

filename = os.path.join(dataroot,filenames[0])
csv_header = read_csv_header(filename)
csv_header = np.array(csv_header)
csv_header = csv_header[important_indices]

count = 0
filtered_count = 0 
for filename in filenames:
    with open(os.path.join(good_flow_dataroot,filename.replace('.csv','_good_flows.txt'))) as f:
        good_flowids = [line.rstrip() for line in f.readlines()] 
    data = read_csv(os.path.join(dataroot,filename))
    data_d = make_dictionary_from_flow(data)

    data_new = extract_n_fix(data_d,good_flowids)
    data_new = np.concatenate((csv_header[np.newaxis,:],data_new),axis=0)    
    print("data shape: {}, data_new.shape: {} ".format(data.shape, data_new.shape))

    count += data.shape[0]
    filtered_count += data_new.shape[0]

    output_filename = os_join(dataroot,filename).replace(dataroot,output_dataroot)
    np.savetxt(output_filename,data_new,fmt="%s",delimiter=",")

print("Filtered flows: {} | Total flows: {}".format(filtered_count,count))
