#in pre processing stage we will split (and write) the data with non-overlapping flowids

import numpy as np
from CICIDS2017 import read_csv
from os.path import join as os_join
import os
import csv

dataroot = r'/media/bek/8E92899D92898A83/arash/CIC-IDS-2017/GeneratedLabelledFlows/TrafficLabelling'
output_folder = r'/media/bek/8E92899D92898A83/arash/CIC-IDS-2017/GeneratedLabelledFlows/TrafficLabellingSplit'




def get_filenames(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isfile(os.path.join(a_dir, name))]


def split_flowids(flow_ids):
    N = len(flowids)
    num_val = N*5//100
    num_test = N*15//100

    indices = np.random.randint(0,N, N)
    permutated_flowids = [flowids[i] for i in indices]
    return (permutated_flowids[:num_val], permutated_flowids[num_val:num_val+num_test], permutated_flowids[num_val+num_test:])


def write_csv(data, filename):
    with open( filename,"w") as csv_file:
        wr = csv.writer(csv_file, quoting=csv.QUOTE_ALL)
        for row in data:
            wr.writerow(row)

def split_and_write_data(data,flowids, input_file):
    val_flowids, test_flowids, train_flowids = split_flowids(flowids)

    val_data = [row for row in data if row[0] in val_flowids]
    write_csv(val_data,input_file.replace("TrafficLabelling", "TrafficLabellingVal"))

    test_data = [row for row in data if row[0] in test_flowids]
    write_csv(test_data,input_file.replace("TrafficLabelling", "TrafficLabellingTest"))

    train_data = [row for row in data if row[0] in train_flowids]
    write_csv(train_data,input_file.replace("TrafficLabelling", "TrafficLabellingTrain"))


filenames = get_filenames(dataroot)

#N is the number of rows
for filename in filenames:
    data = read_csv(os_join(dataroot,filename), sampling_rate=100)
    flowids = [row[0] for row in data] #O(N)
    sorted_index = np.argsort(flowids) # O(NlogN)
    flowids = sorted(flowids)
    split_and_write_data(data, flowids, os_join(dataroot,filename))
    unique, counts = np.unique(flowids, return_counts=True)
    print (np.asarray((unique[:5], counts[:5])).T)

