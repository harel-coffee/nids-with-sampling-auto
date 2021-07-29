#in pre processing stage we will split (and write) the data with non-overlapping flowids

import numpy as np
from CICIDS2017 import read_csv, read_csv_header
from os.path import join as os_join
import os
import csv
from tqdm import tqdm
from collections import defaultdict

#dataroot = r'/home/isrl/data/CIC-IDS-2017/GeneratedLabelledFlows/TrafficLabelling'
dataroot = '/media/bek/8E92899D92898A83/arash/CIC-IDS-2017/PCAPs/sr10'

def get_filenames(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isfile(os.path.join(a_dir, name)) and not name.startswith(".~lock.")]


def get_index_till(counts,sum):
    s = 0
    for i, e in enumerate(counts):
        s+=e
        if s>=sum:
            return i+1



def split_flowids(flowids):
    N = len(flowids)
    num_val = N*5//100
    num_test = N*15//100

    indices = np.random.randint(0,N, N)
    permutated_flowids = [flowids[i] for i in indices]

    unique_flow_ids, counts = np.unique(flowids, return_counts=True)
    val_index = get_index_till(counts,num_val)
    val_flow_ids = unique_flow_ids[:val_index]

    test_val_index = get_index_till(counts,num_val+num_test)
    test_flow_ids = unique_flow_ids[val_index:test_val_index]
    train_flow_ids = unique_flow_ids[test_val_index:]    

    #print (np.asarray((unique[:5], counts[:5])).T)
    return (list(val_flow_ids), list(test_flow_ids), list(train_flow_ids))


def write_csv_d(csv_header,data, filename):
    print("writing to {}".format(filename))
     
    for vals in tqdm(data):
        vals = np.array(vals)
        if (len(set(vals[:,-1])))>1 and len(set(vals[:,6]))>1:
            print(len(set(vals[:,-1])))
            print(vals[:,-1],vals[:,6])


def make_dictionary_from_flow(data):
    d = defaultdict(list)
    for row in data:
         d[row[0]].append(row)
    return d


def split_and_write_data(csv_header, data_d,flowids, input_file):
    val_flowids, test_flowids, train_flowids = split_flowids(flowids)
    print(type(flowids),type(val_flowids), type(val_flowids[0]))    
    val_data = [data_d[str(flow_id)] for flow_id in val_flowids]
    write_csv_d(csv_header,val_data,input_file.replace("TrafficLabelling", "TrafficLabellingVal"))

    test_data = [data_d[flow_id] for flow_id in test_flowids]
    write_csv_d(csv_header,test_data,input_file.replace("TrafficLabelling", "TrafficLabellingTest"))

    train_data = [data_d[flow_id] for flow_id in train_flowids]
    write_csv_d(csv_header, train_data,input_file.replace("TrafficLabelling", "TrafficLabellingTrain"))


filenames = get_filenames(dataroot)

#N is the number of rows
for filename in filenames:
    csv_header = read_csv_header(os_join(dataroot,filename))
    data = read_csv(os_join(dataroot,filename), sampling_rate=10)
    flowids = [row[0] for row in data]
    data_d = make_dictionary_from_flow(data)
    print("#flows: {}".format(len(flowids)))
    split_and_write_data(csv_header,data_d, flowids, os_join(dataroot,filename))

