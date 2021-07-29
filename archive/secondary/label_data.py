import numpy as np
from collections import defaultdict
import os
from os.path import join as os_join
from utils import read_csv_str, read_csv_header
from tqdm import tqdm
import glob
import pandas as pd
from multiprocessing import Pool, Manager, Lock, Process

labelled_dataroot = '/home/juma/data/net_intrusion/CIC-IDS-2018/CSVs/cicflowmeter'
dataroot = '/home/juma/data/net_intrusion/CIC-IDS-2018/CSVs/ffs_(4,8,8)'
outputroot = dataroot + '_l'
if not os.path.exists(outputroot):
    os.mkdir(outputroot)


def make_dictionary_from_flow(data):
    d = defaultdict(list)
    print('making dictionary from flow')
    flowid = None
    for row in tqdm(data.itertuples()):
        # leave 0 for pd index, 1 for flowid,
        flowid = row[2]+'-'+row[3]+'-'+row[4]+'-'+row[5]+'-'+row[6]
        d[flowid].append(row)
    return d


def get_filenames(a_dir):
    if '2018' in a_dir:
        filenames = [f for f in glob.iglob(os_join(a_dir,'*_TrafficForML_CICFlowMeter.csv'))]
    else:
        filenames =  [f for f in glob.iglob(os_join(a_dir,'*.pcap_Flow.csv'))]
    return filenames


#currently used for saving label_dist
def save_dict_to_csv(filename,d):
    with open(filename,'w') as f:
        for key in sorted(d.keys()):
            f.write('{},{}\n'.format(key,d[key]))
        

def label_data(filename,label_dist,lock):
    non_labeled_flow_dist = defaultdict(lambda: 0)
    local_label_dist = defaultdict(lambda: 0)

    data = read_csv_str(filename)
    data_d = make_dictionary_from_flow(data) 
    labelled_data = read_csv_str(filename.replace(dataroot,labelled_dataroot))
    labelled_data_d = make_dictionary_from_flow(labelled_data)
    print(list(labelled_data_d.keys())[0])
    print(list(data_d.keys())[0])

    #debug
    new_data_header = read_csv_header(filename) + ['Label']

    multilabel = 0
    no_labelled_data = 0
    offset=0
    l = []
    for row in data.itertuples():
        #row = data[i]
        flowid = row[2]+'-'+row[3]+'-'+row[4]+'-'+row[5]+'-'+row[6]
        labelled_flow = labelled_data_d[flowid]
        if len(labelled_flow)<1:
            #print("no found",flowid)
            no_labelled_data+=1
            if "output" in dataroot:
                offset = 1
            num_pkt = int(float(row[6+offset]))
            non_labeled_flow_dist[num_pkt]+=1
            continue
        labels = np.array(labelled_flow)[:,-1]
        if len(set(labels))>1:
            multilabel+=1
            continue
        else:
            label = labels[0] # there is only one label
            row = np.concatenate((row,[label]))
            local_label_dist[label]+=1

            l.append(row[1:])
    new_data = pd.DataFrame(l,columns=new_data_header)
    print(new_data.iloc[0,:])
    print('new_data.shape = ',new_data.shape)
    print("Multi labels: {}".format(multilabel))        
    print("non labelled flows: {}".format(no_labelled_data))
    print(new_data.head())
    
    save_dict_to_csv(os_join(dataroot,'non_labeled_flow_length_dist.csv'),non_labeled_flow_dist)
    new_data.to_csv(filename.replace(dataroot,outputroot),index=False, encoding='utf-8-sig')

    #updating shared variable to count label distribution
    with lock:
        for key in local_label_dist.keys():
            if key in label_dist:
                label_dist[key]+=local_label_dist[key]
            else:
                label_dist[key]=local_label_dist[key]


output_filename = os_join(outputroot,"percentiles_for_num_of_packets_forward.csv")
if os.path.exists(output_filename):
    os.remove(output_filename)
filenames = get_filenames(dataroot)
print(filenames)

with Manager() as manager:
    label_dist = manager.dict()
    lock = Lock()
    procs = [Process(target=label_data, args=(filename,label_dist,lock)) for filename in filenames]
    
    for p in procs: p.start()
    for p in procs: p.join()

    print(label_dist)
    
    save_dict_to_csv(os_join(outputroot,'label_dist.csv'),label_dist)
