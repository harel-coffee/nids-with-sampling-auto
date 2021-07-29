import numpy as np
from os.path import join as os_join
import os
from tqdm import tqdm
from collections import defaultdict
from utils import read_csv

#dataroot = '/home/isrl/data/CIC-IDS-2017/PCAPs/check_direction'
dataroot = '/home/isrl/data/CIC-IDS-2017/GeneratedLabelledFlows/TrafficLabellingMerged'

def make_dictionary_from_flow(data):
    d = defaultdict(list)
    print('making dictionary from a flow')
    for row in tqdm(data):
        flowid = row[1]+'-'+row[2]+'-'+row[3]+'-'+row[4]+'-'+row[5]
        d[flowid].append(row)
    return d


def remove_symmetry_n_duplicates(data_d):
    print("Finding good flows by removing symmetric and duplicate flows")
    good_flowids = []
    for key,vals in tqdm(data_d.items()):
        src_ip,src_port,dst_ip,dst_port,protocol = key.split('-')
        reverse_flowid = dst_ip+'-'+dst_port+'-'+src_ip+'-'+src_port+'-'+protocol
        vals = np.array(vals)
        labels = vals[:,-1]
        if reverse_flowid not in data_d and len(set(labels))==1:
            good_flowids.append(key)
    return good_flowids
        

def write_to_file(filename,flowids):
    print('Writing good flows to file')
    with open(filename,'w') as f:    
        for flowid in flowids:
            f.write("{}\n".format(flowid))


def get_filenames(a_dir):
    return [name for name in os.listdir(a_dir) if os.path.isfile(os.path.join(a_dir,name)) and not name.startswith(".~lock.") and (name.endswith(".pcap_ISCX.csv") or name.endswith(".pcap_Flow.csv"))]


flow_count = 0
good_flow_count = 0
filenames = get_filenames(dataroot)

for filename in filenames:
    filename = os.path.join(dataroot,filename)
    data_i = read_csv(filename)
    flow_count+=data_i.shape[0]

    data_d = make_dictionary_from_flow(data_i)
    good_flowids = remove_symmetry_n_duplicates(data_d)
    good_flow_count += len(good_flowids)
    write_to_file(filename.replace('.csv','_good_flows.txt'),good_flowids)
    print("{:50}: entry {}| dict {}| good flows {}".format(filename,data_i.shape[0],len(data_d),len(good_flowids)))

print("Summary: good flow count {:10d}| flow count {:10d}".format(good_flow_count,flow_count))


